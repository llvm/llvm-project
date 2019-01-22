//===--- ClangdUnit.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdUnit.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "../clang-tidy/ClangTidyModuleRegistry.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "Logger.h"
#include "SourceCode.h"
#include "Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

bool compileCommandsAreEqual(const tooling::CompileCommand &LHS,
                             const tooling::CompileCommand &RHS) {
  // We don't check for Output, it should not matter to clangd.
  return LHS.Directory == RHS.Directory && LHS.Filename == RHS.Filename &&
         llvm::makeArrayRef(LHS.CommandLine).equals(RHS.CommandLine);
}

template <class T> std::size_t getUsedBytes(const std::vector<T> &Vec) {
  return Vec.capacity() * sizeof(T);
}

class DeclTrackingASTConsumer : public ASTConsumer {
public:
  DeclTrackingASTConsumer(std::vector<Decl *> &TopLevelDecls)
      : TopLevelDecls(TopLevelDecls) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
      if (D->isFromASTFile())
        continue;

      // ObjCMethodDecl are not actually top-level decls.
      if (isa<ObjCMethodDecl>(D))
        continue;

      TopLevelDecls.push_back(D);
    }
    return true;
  }

private:
  std::vector<Decl *> &TopLevelDecls;
};

class ClangdFrontendAction : public SyntaxOnlyAction {
public:
  std::vector<Decl *> takeTopLevelDecls() { return std::move(TopLevelDecls); }

protected:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) override {
    return llvm::make_unique<DeclTrackingASTConsumer>(/*ref*/ TopLevelDecls);
  }

private:
  std::vector<Decl *> TopLevelDecls;
};

class CppFilePreambleCallbacks : public PreambleCallbacks {
public:
  CppFilePreambleCallbacks(PathRef File, PreambleParsedCallback ParsedCallback)
      : File(File), ParsedCallback(ParsedCallback) {}

  IncludeStructure takeIncludes() { return std::move(Includes); }

  void AfterExecute(CompilerInstance &CI) override {
    if (!ParsedCallback)
      return;
    trace::Span Tracer("Running PreambleCallback");
    ParsedCallback(CI.getASTContext(), CI.getPreprocessorPtr());
  }

  void BeforeExecute(CompilerInstance &CI) override {
    SourceMgr = &CI.getSourceManager();
  }

  std::unique_ptr<PPCallbacks> createPPCallbacks() override {
    assert(SourceMgr && "SourceMgr must be set at this point");
    return collectIncludeStructureCallback(*SourceMgr, &Includes);
  }

private:
  PathRef File;
  PreambleParsedCallback ParsedCallback;
  IncludeStructure Includes;
  SourceManager *SourceMgr = nullptr;
};

// When using a preamble, only preprocessor events outside its bounds are seen.
// This is almost what we want: replaying transitive preprocessing wastes time.
// However this confuses clang-tidy checks: they don't see any #includes!
// So we replay the *non-transitive* #includes that appear in the main-file.
// It would be nice to replay other events (macro definitions, ifdefs etc) but
// this addresses the most common cases fairly cheaply.
class ReplayPreamble : private PPCallbacks {
public:
  // Attach preprocessor hooks such that preamble events will be injected at
  // the appropriate time.
  // Events will be delivered to the *currently registered* PP callbacks.
  static void attach(const IncludeStructure &Includes,
                     CompilerInstance &Clang) {
    auto &PP = Clang.getPreprocessor();
    auto *ExistingCallbacks = PP.getPPCallbacks();
    // No need to replay events if nobody is listening.
    if (!ExistingCallbacks)
      return;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(
        new ReplayPreamble(Includes, ExistingCallbacks,
                           Clang.getSourceManager(), PP, Clang.getLangOpts())));
    // We're relying on the fact that addPPCallbacks keeps the old PPCallbacks
    // around, creating a chaining wrapper. Guard against other implementations.
    assert(PP.getPPCallbacks() != ExistingCallbacks &&
           "Expected chaining implementation");
  }

private:
  ReplayPreamble(const IncludeStructure &Includes, PPCallbacks *Delegate,
                 const SourceManager &SM, Preprocessor &PP,
                 const LangOptions &LangOpts)
      : Includes(Includes), Delegate(Delegate), SM(SM), PP(PP),
        LangOpts(LangOpts) {}

  // In a normal compile, the preamble traverses the following structure:
  //
  // mainfile.cpp
  //   <built-in>
  //     ... macro definitions like __cplusplus ...
  //     <command-line>
  //       ... macro definitions for args like -Dfoo=bar ...
  //   "header1.h"
  //     ... header file contents ...
  //   "header2.h"
  //     ... header file contents ...
  //   ... main file contents ...
  //
  // When using a preamble, the "header1" and "header2" subtrees get skipped.
  // We insert them right after the built-in header, which still appears.
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind Kind, FileID PrevFID) override {
    // It'd be nice if there was a better way to identify built-in headers...
    if (Reason == FileChangeReason::ExitFile &&
        SM.getBuffer(PrevFID)->getBufferIdentifier() == "<built-in>")
      replay();
  }

  void replay() {
    for (const auto &Inc : Includes.MainFileIncludes) {
      const FileEntry *File = nullptr;
      if (Inc.Resolved != "")
        File = SM.getFileManager().getFile(Inc.Resolved);

      llvm::StringRef WrittenFilename =
          llvm::StringRef(Inc.Written).drop_front().drop_back();
      bool Angled = llvm::StringRef(Inc.Written).startswith("<");

      // Re-lex the #include directive to find its interesting parts.
      llvm::StringRef Src = SM.getBufferData(SM.getMainFileID());
      Lexer RawLexer(SM.getLocForStartOfFile(SM.getMainFileID()), LangOpts,
                     Src.begin(), Src.begin() + Inc.HashOffset, Src.end());
      Token HashTok, IncludeTok, FilenameTok;
      RawLexer.LexFromRawLexer(HashTok);
      assert(HashTok.getKind() == tok::hash);
      RawLexer.setParsingPreprocessorDirective(true);
      RawLexer.LexFromRawLexer(IncludeTok);
      IdentifierInfo *II = PP.getIdentifierInfo(IncludeTok.getRawIdentifier());
      IncludeTok.setIdentifierInfo(II);
      IncludeTok.setKind(II->getTokenID());
      RawLexer.LexIncludeFilename(FilenameTok);

      Delegate->InclusionDirective(
          HashTok.getLocation(), IncludeTok, WrittenFilename, Angled,
          CharSourceRange::getCharRange(FilenameTok.getLocation(),
                                        FilenameTok.getEndLoc()),
          File, "SearchPath", "RelPath", /*Imported=*/nullptr, Inc.FileKind);
      if (File)
        Delegate->FileSkipped(*File, FilenameTok, Inc.FileKind);
      else {
        llvm::SmallString<1> UnusedRecovery;
        Delegate->FileNotFound(WrittenFilename, UnusedRecovery);
      }
    }
  }

  const IncludeStructure &Includes;
  PPCallbacks *Delegate;
  const SourceManager &SM;
  Preprocessor &PP;
  const LangOptions &LangOpts;
};

} // namespace

void dumpAST(ParsedAST &AST, llvm::raw_ostream &OS) {
  AST.getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

llvm::Optional<ParsedAST>
ParsedAST::build(std::unique_ptr<CompilerInvocation> CI,
                 std::shared_ptr<const PreambleData> Preamble,
                 std::unique_ptr<llvm::MemoryBuffer> Buffer,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                 const tidy::ClangTidyOptions &ClangTidyOpts) {
  assert(CI);
  // Command-line parsing sets DisableFree to true by default, but we don't want
  // to leak memory in clangd.
  CI->getFrontendOpts().DisableFree = false;
  const PrecompiledPreamble *PreamblePCH =
      Preamble ? &Preamble->Preamble : nullptr;

  StoreDiags ASTDiags;
  auto Clang =
      prepareCompilerInstance(std::move(CI), PreamblePCH, std::move(Buffer),
                              std::move(PCHs), std::move(VFS), ASTDiags);
  if (!Clang)
    return None;

  auto Action = llvm::make_unique<ClangdFrontendAction>();
  const FrontendInputFile &MainInput = Clang->getFrontendOpts().Inputs[0];
  if (!Action->BeginSourceFile(*Clang, MainInput)) {
    log("BeginSourceFile() failed when building AST for {0}",
        MainInput.getFile());
    return None;
  }

  // Set up ClangTidy. Must happen after BeginSourceFile() so ASTContext exists.
  // Clang-tidy has some limitiations to ensure reasonable performance:
  //  - checks don't see all preprocessor events in the preamble
  //  - matchers run only over the main-file top-level decls (and can't see
  //    ancestors outside this scope).
  // In practice almost all checks work well without modifications.
  std::vector<std::unique_ptr<tidy::ClangTidyCheck>> CTChecks;
  ast_matchers::MatchFinder CTFinder;
  llvm::Optional<tidy::ClangTidyContext> CTContext;
  {
    trace::Span Tracer("ClangTidyInit");
    tidy::ClangTidyCheckFactories CTFactories;
    for (const auto &E : tidy::ClangTidyModuleRegistry::entries())
      E.instantiate()->addCheckFactories(CTFactories);
    CTContext.emplace(llvm::make_unique<tidy::DefaultOptionsProvider>(
        tidy::ClangTidyGlobalOptions(), ClangTidyOpts));
    CTContext->setDiagnosticsEngine(&Clang->getDiagnostics());
    CTContext->setASTContext(&Clang->getASTContext());
    CTContext->setCurrentFile(MainInput.getFile());
    CTFactories.createChecks(CTContext.getPointer(), CTChecks);
    for (const auto &Check : CTChecks) {
      // FIXME: the PP callbacks skip the entire preamble.
      // Checks that want to see #includes in the main file do not see them.
      Check->registerPPCallbacks(*Clang);
      Check->registerMatchers(&CTFinder);
    }
  }

  // Copy over the includes from the preamble, then combine with the
  // non-preamble includes below.
  auto Includes = Preamble ? Preamble->Includes : IncludeStructure{};
  // Replay the preamble includes so that clang-tidy checks can see them.
  if (Preamble)
    ReplayPreamble::attach(Includes, *Clang);
  // Important: collectIncludeStructure is registered *after* ReplayPreamble!
  // Otherwise we would collect the replayed includes again...
  // (We can't *just* use the replayed includes, they don't have Resolved path).
  Clang->getPreprocessor().addPPCallbacks(
      collectIncludeStructureCallback(Clang->getSourceManager(), &Includes));

  if (!Action->Execute())
    log("Execute() failed when building AST for {0}", MainInput.getFile());

  std::vector<Decl *> ParsedDecls = Action->takeTopLevelDecls();
  // AST traversals should exclude the preamble, to avoid performance cliffs.
  Clang->getASTContext().setTraversalScope(ParsedDecls);
  {
    // Run the AST-dependent part of the clang-tidy checks.
    // (The preprocessor part ran already, via PPCallbacks).
    trace::Span Tracer("ClangTidyMatch");
    CTFinder.matchAST(Clang->getASTContext());
  }

  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new IgnoreDiagnostics);
  // CompilerInstance won't run this callback, do it directly.
  ASTDiags.EndSourceFile();
  // XXX: This is messy: clang-tidy checks flush some diagnostics at EOF.
  // However Action->EndSourceFile() would destroy the ASTContext!
  // So just inform the preprocessor of EOF, while keeping everything alive.
  Clang->getPreprocessor().EndSourceFile();

  std::vector<Diag> Diags = ASTDiags.take();
  // Add diagnostics from the preamble, if any.
  if (Preamble)
    Diags.insert(Diags.begin(), Preamble->Diags.begin(), Preamble->Diags.end());
  return ParsedAST(std::move(Preamble), std::move(Clang), std::move(Action),
                   std::move(ParsedDecls), std::move(Diags),
                   std::move(Includes));
}

ParsedAST::ParsedAST(ParsedAST &&Other) = default;

ParsedAST &ParsedAST::operator=(ParsedAST &&Other) = default;

ParsedAST::~ParsedAST() {
  if (Action) {
    // We already notified the PP of end-of-file earlier, so detach it first.
    // We must keep it alive until after EndSourceFile(), Sema relies on this.
    auto PP = Clang->getPreprocessorPtr(); // Keep PP alive for now.
    Clang->setPreprocessor(nullptr);       // Detach so we don't send EOF again.
    Action->EndSourceFile();               // Destroy ASTContext and Sema.
    // Now Sema is gone, it's safe for PP to go out of scope.
  }
}

ASTContext &ParsedAST::getASTContext() { return Clang->getASTContext(); }

const ASTContext &ParsedAST::getASTContext() const {
  return Clang->getASTContext();
}

Preprocessor &ParsedAST::getPreprocessor() { return Clang->getPreprocessor(); }

std::shared_ptr<Preprocessor> ParsedAST::getPreprocessorPtr() {
  return Clang->getPreprocessorPtr();
}

const Preprocessor &ParsedAST::getPreprocessor() const {
  return Clang->getPreprocessor();
}

llvm::ArrayRef<Decl *> ParsedAST::getLocalTopLevelDecls() {
  return LocalTopLevelDecls;
}

const std::vector<Diag> &ParsedAST::getDiagnostics() const { return Diags; }

std::size_t ParsedAST::getUsedBytes() const {
  auto &AST = getASTContext();
  // FIXME(ibiryukov): we do not account for the dynamically allocated part of
  // Message and Fixes inside each diagnostic.
  std::size_t Total =
      clangd::getUsedBytes(LocalTopLevelDecls) + clangd::getUsedBytes(Diags);

  // FIXME: the rest of the function is almost a direct copy-paste from
  // libclang's clang_getCXTUResourceUsage. We could share the implementation.

  // Sum up variaous allocators inside the ast context and the preprocessor.
  Total += AST.getASTAllocatedMemory();
  Total += AST.getSideTableAllocatedMemory();
  Total += AST.Idents.getAllocator().getTotalMemory();
  Total += AST.Selectors.getTotalMemory();

  Total += AST.getSourceManager().getContentCacheSize();
  Total += AST.getSourceManager().getDataStructureSizes();
  Total += AST.getSourceManager().getMemoryBufferSizes().malloc_bytes;

  if (ExternalASTSource *Ext = AST.getExternalSource())
    Total += Ext->getMemoryBufferSizes().malloc_bytes;

  const Preprocessor &PP = getPreprocessor();
  Total += PP.getTotalMemory();
  if (PreprocessingRecord *PRec = PP.getPreprocessingRecord())
    Total += PRec->getTotalMemory();
  Total += PP.getHeaderSearchInfo().getTotalMemory();

  return Total;
}

const IncludeStructure &ParsedAST::getIncludeStructure() const {
  return Includes;
}

PreambleData::PreambleData(PrecompiledPreamble Preamble,
                           std::vector<Diag> Diags, IncludeStructure Includes,
                           std::unique_ptr<PreambleFileStatusCache> StatCache)
    : Preamble(std::move(Preamble)), Diags(std::move(Diags)),
      Includes(std::move(Includes)), StatCache(std::move(StatCache)) {}

ParsedAST::ParsedAST(std::shared_ptr<const PreambleData> Preamble,
                     std::unique_ptr<CompilerInstance> Clang,
                     std::unique_ptr<FrontendAction> Action,
                     std::vector<Decl *> LocalTopLevelDecls,
                     std::vector<Diag> Diags, IncludeStructure Includes)
    : Preamble(std::move(Preamble)), Clang(std::move(Clang)),
      Action(std::move(Action)), Diags(std::move(Diags)),
      LocalTopLevelDecls(std::move(LocalTopLevelDecls)),
      Includes(std::move(Includes)) {
  assert(this->Clang);
  assert(this->Action);
}

std::shared_ptr<const PreambleData>
buildPreamble(PathRef FileName, CompilerInvocation &CI,
              std::shared_ptr<const PreambleData> OldPreamble,
              const tooling::CompileCommand &OldCompileCommand,
              const ParseInputs &Inputs,
              std::shared_ptr<PCHContainerOperations> PCHs, bool StoreInMemory,
              PreambleParsedCallback PreambleCallback) {
  // Note that we don't need to copy the input contents, preamble can live
  // without those.
  auto ContentsBuffer = llvm::MemoryBuffer::getMemBuffer(Inputs.Contents);
  auto Bounds =
      ComputePreambleBounds(*CI.getLangOpts(), ContentsBuffer.get(), 0);

  if (OldPreamble &&
      compileCommandsAreEqual(Inputs.CompileCommand, OldCompileCommand) &&
      OldPreamble->Preamble.CanReuse(CI, ContentsBuffer.get(), Bounds,
                                     Inputs.FS.get())) {
    vlog("Reusing preamble for file {0}", llvm::Twine(FileName));
    return OldPreamble;
  }
  vlog("Preamble for file {0} cannot be reused. Attempting to rebuild it.",
       FileName);

  trace::Span Tracer("BuildPreamble");
  SPAN_ATTACH(Tracer, "File", FileName);
  StoreDiags PreambleDiagnostics;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
      CompilerInstance::createDiagnostics(&CI.getDiagnosticOpts(),
                                          &PreambleDiagnostics, false);

  // Skip function bodies when building the preamble to speed up building
  // the preamble and make it smaller.
  assert(!CI.getFrontendOpts().SkipFunctionBodies);
  CI.getFrontendOpts().SkipFunctionBodies = true;
  // We don't want to write comment locations into PCH. They are racy and slow
  // to read back. We rely on dynamic index for the comments instead.
  CI.getPreprocessorOpts().WriteCommentListToPCH = false;

  CppFilePreambleCallbacks SerializedDeclsCollector(FileName, PreambleCallback);
  if (Inputs.FS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory)) {
    log("Couldn't set working directory when building the preamble.");
    // We proceed anyway, our lit-tests rely on results for non-existing working
    // dirs.
  }

  llvm::SmallString<32> AbsFileName(FileName);
  Inputs.FS->makeAbsolute(AbsFileName);
  auto StatCache = llvm::make_unique<PreambleFileStatusCache>(AbsFileName);
  auto BuiltPreamble = PrecompiledPreamble::Build(
      CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine,
      StatCache->getProducingFS(Inputs.FS), PCHs, StoreInMemory,
      SerializedDeclsCollector);

  // When building the AST for the main file, we do want the function
  // bodies.
  CI.getFrontendOpts().SkipFunctionBodies = false;

  if (BuiltPreamble) {
    vlog("Built preamble of size {0} for file {1}", BuiltPreamble->getSize(),
         FileName);
    return std::make_shared<PreambleData>(
        std::move(*BuiltPreamble), PreambleDiagnostics.take(),
        SerializedDeclsCollector.takeIncludes(), std::move(StatCache));
  } else {
    elog("Could not build a preamble for file {0}", FileName);
    return nullptr;
  }
}

llvm::Optional<ParsedAST>
buildAST(PathRef FileName, std::unique_ptr<CompilerInvocation> Invocation,
         const ParseInputs &Inputs,
         std::shared_ptr<const PreambleData> Preamble,
         std::shared_ptr<PCHContainerOperations> PCHs) {
  trace::Span Tracer("BuildAST");
  SPAN_ATTACH(Tracer, "File", FileName);

  auto VFS = Inputs.FS;
  if (Preamble && Preamble->StatCache)
    VFS = Preamble->StatCache->getConsumingFS(std::move(VFS));
  if (VFS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory)) {
    log("Couldn't set working directory when building the preamble.");
    // We proceed anyway, our lit-tests rely on results for non-existing working
    // dirs.
  }

  return ParsedAST::build(llvm::make_unique<CompilerInvocation>(*Invocation),
                          Preamble,
                          llvm::MemoryBuffer::getMemBufferCopy(Inputs.Contents),
                          PCHs, std::move(VFS), Inputs.ClangTidyOpts);
}

SourceLocation getBeginningOfIdentifier(ParsedAST &Unit, const Position &Pos,
                                        const FileID FID) {
  const ASTContext &AST = Unit.getASTContext();
  const SourceManager &SourceMgr = AST.getSourceManager();
  auto Offset = positionToOffset(SourceMgr.getBufferData(FID), Pos);
  if (!Offset) {
    log("getBeginningOfIdentifier: {0}", Offset.takeError());
    return SourceLocation();
  }

  // GetBeginningOfToken(pos) is almost what we want, but does the wrong thing
  // if the cursor is at the end of the identifier.
  // Instead, we lex at GetBeginningOfToken(pos - 1). The cases are:
  //  1) at the beginning of an identifier, we'll be looking at something
  //  that isn't an identifier.
  //  2) at the middle or end of an identifier, we get the identifier.
  //  3) anywhere outside an identifier, we'll get some non-identifier thing.
  // We can't actually distinguish cases 1 and 3, but returning the original
  // location is correct for both!
  SourceLocation InputLoc = SourceMgr.getComposedLoc(FID, *Offset);
  if (*Offset == 0) // Case 1 or 3.
    return SourceMgr.getMacroArgExpandedLocation(InputLoc);
  SourceLocation Before = SourceMgr.getComposedLoc(FID, *Offset - 1);

  Before = Lexer::GetBeginningOfToken(Before, SourceMgr, AST.getLangOpts());
  Token Tok;
  if (Before.isValid() &&
      !Lexer::getRawToken(Before, Tok, SourceMgr, AST.getLangOpts(), false) &&
      Tok.is(tok::raw_identifier))
    return SourceMgr.getMacroArgExpandedLocation(Before); // Case 2.
  return SourceMgr.getMacroArgExpandedLocation(InputLoc); // Case 1 or 3.
}

} // namespace clangd
namespace tidy {
// Force the linker to link in Clang-tidy modules.
#define LINK_TIDY_MODULE(X)                                                    \
  extern volatile int X##ModuleAnchorSource;                                   \
  static int LLVM_ATTRIBUTE_UNUSED X##ModuleAnchorDestination =                \
      X##ModuleAnchorSource
LINK_TIDY_MODULE(CERT);
LINK_TIDY_MODULE(Abseil);
LINK_TIDY_MODULE(Boost);
LINK_TIDY_MODULE(Bugprone);
LINK_TIDY_MODULE(LLVM);
LINK_TIDY_MODULE(CppCoreGuidelines);
LINK_TIDY_MODULE(Fuchsia);
LINK_TIDY_MODULE(Google);
LINK_TIDY_MODULE(Android);
LINK_TIDY_MODULE(Misc);
LINK_TIDY_MODULE(Modernize);
LINK_TIDY_MODULE(Performance);
LINK_TIDY_MODULE(Portability);
LINK_TIDY_MODULE(Readability);
LINK_TIDY_MODULE(ObjC);
LINK_TIDY_MODULE(HICPP);
LINK_TIDY_MODULE(Zircon);
#undef LINK_TIDY_MODULE
} // namespace tidy
} // namespace clang
