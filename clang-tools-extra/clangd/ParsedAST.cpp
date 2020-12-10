//===--- ParsedAST.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "../clang-tidy/ClangTidyCheck.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "../clang-tidy/ClangTidyModuleRegistry.h"
#include "AST.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "IncludeFixer.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "TidyProvider.h"
#include "index/CanonicalIncludes.h"
#include "index/Index.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <vector>

// Force the linker to link in Clang-tidy modules.
// clangd doesn't support the static analyzer.
#define CLANG_TIDY_DISABLE_STATIC_ANALYZER_CHECKS
#include "../clang-tidy/ClangTidyForceLinker.h"

namespace clang {
namespace clangd {
namespace {

template <class T> std::size_t getUsedBytes(const std::vector<T> &Vec) {
  return Vec.capacity() * sizeof(T);
}

class DeclTrackingASTConsumer : public ASTConsumer {
public:
  DeclTrackingASTConsumer(std::vector<Decl *> &TopLevelDecls)
      : TopLevelDecls(TopLevelDecls) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
      auto &SM = D->getASTContext().getSourceManager();
      if (!isInsideMainFile(D->getLocation(), SM))
        continue;
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
        if (isImplicitTemplateInstantiation(ND))
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
    return std::make_unique<DeclTrackingASTConsumer>(/*ref*/ TopLevelDecls);
  }

private:
  std::vector<Decl *> TopLevelDecls;
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
  static void attach(std::vector<Inclusion> Includes, CompilerInstance &Clang,
                     const PreambleBounds &PB) {
    auto &PP = Clang.getPreprocessor();
    auto *ExistingCallbacks = PP.getPPCallbacks();
    // No need to replay events if nobody is listening.
    if (!ExistingCallbacks)
      return;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(new ReplayPreamble(
        std::move(Includes), ExistingCallbacks, Clang.getSourceManager(), PP,
        Clang.getLangOpts(), PB)));
    // We're relying on the fact that addPPCallbacks keeps the old PPCallbacks
    // around, creating a chaining wrapper. Guard against other implementations.
    assert(PP.getPPCallbacks() != ExistingCallbacks &&
           "Expected chaining implementation");
  }

private:
  ReplayPreamble(std::vector<Inclusion> Includes, PPCallbacks *Delegate,
                 const SourceManager &SM, Preprocessor &PP,
                 const LangOptions &LangOpts, const PreambleBounds &PB)
      : Includes(std::move(Includes)), Delegate(Delegate), SM(SM), PP(PP) {
    // Only tokenize the preamble section of the main file, as we are not
    // interested in the rest of the tokens.
    MainFileTokens = syntax::tokenize(
        syntax::FileRange(SM.getMainFileID(), 0, PB.Size), SM, LangOpts);
  }

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
        SM.getBufferOrFake(PrevFID).getBufferIdentifier() == "<built-in>")
      replay();
  }

  void replay() {
    for (const auto &Inc : Includes) {
      llvm::Optional<FileEntryRef> File;
      if (Inc.Resolved != "")
        File = expectedToOptional(SM.getFileManager().getFileRef(Inc.Resolved));

      // Re-lex the #include directive to find its interesting parts.
      auto HashLoc = SM.getComposedLoc(SM.getMainFileID(), Inc.HashOffset);
      auto HashTok = llvm::partition_point(MainFileTokens,
                                           [&HashLoc](const syntax::Token &T) {
                                             return T.location() < HashLoc;
                                           });
      assert(HashTok != MainFileTokens.end() && HashTok->kind() == tok::hash);

      auto IncludeTok = std::next(HashTok);
      assert(IncludeTok != MainFileTokens.end());

      auto FileTok = std::next(IncludeTok);
      assert(FileTok != MainFileTokens.end());

      // Create a fake import/include token, none of the callers seem to care
      // about clang::Token::Flags.
      Token SynthesizedIncludeTok;
      SynthesizedIncludeTok.startToken();
      SynthesizedIncludeTok.setLocation(IncludeTok->location());
      SynthesizedIncludeTok.setLength(IncludeTok->length());
      SynthesizedIncludeTok.setKind(tok::raw_identifier);
      SynthesizedIncludeTok.setRawIdentifierData(IncludeTok->text(SM).data());
      PP.LookUpIdentifierInfo(SynthesizedIncludeTok);

      // Same here, create a fake one for Filename, including angles or quotes.
      Token SynthesizedFilenameTok;
      SynthesizedFilenameTok.startToken();
      SynthesizedFilenameTok.setLocation(FileTok->location());
      // Note that we can't make use of FileTok->length/text in here as in the
      // case of angled includes this will contain tok::less instead of
      // filename. Whereas Inc.Written contains the full header name including
      // quotes/angles.
      SynthesizedFilenameTok.setLength(Inc.Written.length());
      SynthesizedFilenameTok.setKind(tok::header_name);
      SynthesizedFilenameTok.setLiteralData(Inc.Written.data());

      const FileEntry *FE = File ? &File->getFileEntry() : nullptr;
      llvm::StringRef WrittenFilename =
          llvm::StringRef(Inc.Written).drop_front().drop_back();
      Delegate->InclusionDirective(HashTok->location(), SynthesizedIncludeTok,
                                   WrittenFilename, Inc.Written.front() == '<',
                                   FileTok->range(SM).toCharRange(SM), FE,
                                   "SearchPath", "RelPath",
                                   /*Imported=*/nullptr, Inc.FileKind);
      if (File)
        Delegate->FileSkipped(*File, SynthesizedFilenameTok, Inc.FileKind);
      else {
        llvm::SmallString<1> UnusedRecovery;
        Delegate->FileNotFound(WrittenFilename, UnusedRecovery);
      }
    }
  }

  const std::vector<Inclusion> Includes;
  PPCallbacks *Delegate;
  const SourceManager &SM;
  Preprocessor &PP;
  std::vector<syntax::Token> MainFileTokens;
};

} // namespace

llvm::Optional<ParsedAST>
ParsedAST::build(llvm::StringRef Filename, const ParseInputs &Inputs,
                 std::unique_ptr<clang::CompilerInvocation> CI,
                 llvm::ArrayRef<Diag> CompilerInvocationDiags,
                 std::shared_ptr<const PreambleData> Preamble) {
  trace::Span Tracer("BuildAST");
  SPAN_ATTACH(Tracer, "File", Filename);

  auto VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  if (Preamble && Preamble->StatCache)
    VFS = Preamble->StatCache->getConsumingFS(std::move(VFS));

  assert(CI);
  // Command-line parsing sets DisableFree to true by default, but we don't want
  // to leak memory in clangd.
  CI->getFrontendOpts().DisableFree = false;
  const PrecompiledPreamble *PreamblePCH =
      Preamble ? &Preamble->Preamble : nullptr;

  // This is on-by-default in windows to allow parsing SDK headers, but it
  // breaks many features. Disable it for the main-file (not preamble).
  CI->getLangOpts()->DelayedTemplateParsing = false;

  StoreDiags ASTDiags;

  llvm::Optional<PreamblePatch> Patch;
  if (Preamble) {
    Patch = PreamblePatch::create(Filename, Inputs, *Preamble);
    Patch->apply(*CI);
  }
  auto Clang = prepareCompilerInstance(
      std::move(CI), PreamblePCH,
      llvm::MemoryBuffer::getMemBufferCopy(Inputs.Contents, Filename), VFS,
      ASTDiags);
  if (!Clang)
    return None;

  auto Action = std::make_unique<ClangdFrontendAction>();
  const FrontendInputFile &MainInput = Clang->getFrontendOpts().Inputs[0];
  if (!Action->BeginSourceFile(*Clang, MainInput)) {
    log("BeginSourceFile() failed when building AST for {0}",
        MainInput.getFile());
    return None;
  }

  // Set up ClangTidy. Must happen after BeginSourceFile() so ASTContext exists.
  // Clang-tidy has some limitations to ensure reasonable performance:
  //  - checks don't see all preprocessor events in the preamble
  //  - matchers run only over the main-file top-level decls (and can't see
  //    ancestors outside this scope).
  // In practice almost all checks work well without modifications.
  std::vector<std::unique_ptr<tidy::ClangTidyCheck>> CTChecks;
  ast_matchers::MatchFinder CTFinder;
  llvm::Optional<tidy::ClangTidyContext> CTContext;
  {
    trace::Span Tracer("ClangTidyInit");
    tidy::ClangTidyOptions ClangTidyOpts =
        getTidyOptionsForFile(Inputs.ClangTidyProvider, Filename);
    dlog("ClangTidy configuration for file {0}: {1}", Filename,
         tidy::configurationAsText(ClangTidyOpts));
    tidy::ClangTidyCheckFactories CTFactories;
    for (const auto &E : tidy::ClangTidyModuleRegistry::entries())
      E.instantiate()->addCheckFactories(CTFactories);
    CTContext.emplace(std::make_unique<tidy::DefaultOptionsProvider>(
        tidy::ClangTidyGlobalOptions(), ClangTidyOpts));
    CTContext->setDiagnosticsEngine(&Clang->getDiagnostics());
    CTContext->setASTContext(&Clang->getASTContext());
    CTContext->setCurrentFile(Filename);
    CTChecks = CTFactories.createChecks(CTContext.getPointer());
    llvm::erase_if(CTChecks, [&](const auto &Check) {
      return !Check->isLanguageVersionSupported(CTContext->getLangOpts());
    });
    Preprocessor *PP = &Clang->getPreprocessor();
    for (const auto &Check : CTChecks) {
      Check->registerPPCallbacks(Clang->getSourceManager(), PP, PP);
      Check->registerMatchers(&CTFinder);
    }

    if (!CTChecks.empty()) {
      ASTDiags.setLevelAdjuster([&CTContext](DiagnosticsEngine::Level DiagLevel,
                                             const clang::Diagnostic &Info) {
        std::string CheckName = CTContext->getCheckName(Info.getID());
        bool IsClangTidyDiag = !CheckName.empty();
        if (IsClangTidyDiag) {
          // Check for suppression comment. Skip the check for diagnostics not
          // in the main file, because we don't want that function to query the
          // source buffer for preamble files. For the same reason, we ask
          // shouldSuppressDiagnostic to avoid I/O.
          // We let suppression comments take precedence over warning-as-error
          // to match clang-tidy's behaviour.
          bool IsInsideMainFile =
              Info.hasSourceManager() &&
              isInsideMainFile(Info.getLocation(), Info.getSourceManager());
          if (IsInsideMainFile &&
              tidy::shouldSuppressDiagnostic(DiagLevel, Info, *CTContext,
                                             /*AllowIO=*/false)) {
            return DiagnosticsEngine::Ignored;
          }

          // Check for warning-as-error.
          if (DiagLevel == DiagnosticsEngine::Warning &&
              CTContext->treatAsError(CheckName)) {
            return DiagnosticsEngine::Error;
          }
        }
        return DiagLevel;
      });
    }
  }

  // Add IncludeFixer which can recover diagnostics caused by missing includes
  // (e.g. incomplete type) and attach include insertion fixes to diagnostics.
  llvm::Optional<IncludeFixer> FixIncludes;
  auto BuildDir = VFS->getCurrentWorkingDirectory();
  if (Inputs.Opts.SuggestMissingIncludes && Inputs.Index &&
      !BuildDir.getError()) {
    auto Style = getFormatStyleForFile(Filename, Inputs.Contents, *Inputs.TFS);
    auto Inserter = std::make_shared<IncludeInserter>(
        Filename, Inputs.Contents, Style, BuildDir.get(),
        &Clang->getPreprocessor().getHeaderSearchInfo());
    if (Preamble) {
      for (const auto &Inc : Preamble->Includes.MainFileIncludes)
        Inserter->addExisting(Inc);
    }
    FixIncludes.emplace(Filename, Inserter, *Inputs.Index,
                        /*IndexRequestLimit=*/5);
    ASTDiags.contributeFixes([&FixIncludes](DiagnosticsEngine::Level DiagLevl,
                                            const clang::Diagnostic &Info) {
      return FixIncludes->fix(DiagLevl, Info);
    });
    Clang->setExternalSemaSource(FixIncludes->unresolvedNameRecorder());
  }

  IncludeStructure Includes;
  // If we are using a preamble, copy existing includes.
  if (Preamble) {
    Includes = Preamble->Includes;
    Includes.MainFileIncludes = Patch->preambleIncludes();
    // Replay the preamble includes so that clang-tidy checks can see them.
    ReplayPreamble::attach(Patch->preambleIncludes(), *Clang,
                           Patch->modifiedBounds());
  }
  // Important: collectIncludeStructure is registered *after* ReplayPreamble!
  // Otherwise we would collect the replayed includes again...
  // (We can't *just* use the replayed includes, they don't have Resolved path).
  Clang->getPreprocessor().addPPCallbacks(
      collectIncludeStructureCallback(Clang->getSourceManager(), &Includes));
  // Copy over the macros in the preamble region of the main file, and combine
  // with non-preamble macros below.
  MainFileMacros Macros;
  if (Preamble)
    Macros = Preamble->Macros;
  Clang->getPreprocessor().addPPCallbacks(
      std::make_unique<CollectMainFileMacros>(Clang->getSourceManager(),
                                              Macros));

  // Copy over the includes from the preamble, then combine with the
  // non-preamble includes below.
  CanonicalIncludes CanonIncludes;
  if (Preamble)
    CanonIncludes = Preamble->CanonIncludes;
  else
    CanonIncludes.addSystemHeadersMapping(Clang->getLangOpts());
  std::unique_ptr<CommentHandler> IWYUHandler =
      collectIWYUHeaderMaps(&CanonIncludes);
  Clang->getPreprocessor().addCommentHandler(IWYUHandler.get());

  // Collect tokens of the main file.
  syntax::TokenCollector CollectTokens(Clang->getPreprocessor());

  if (llvm::Error Err = Action->Execute())
    log("Execute() failed when building AST for {0}: {1}", MainInput.getFile(),
        toString(std::move(Err)));

  // We have to consume the tokens before running clang-tidy to avoid collecting
  // tokens from running the preprocessor inside the checks (only
  // modernize-use-trailing-return-type does that today).
  syntax::TokenBuffer Tokens = std::move(CollectTokens).consume();
  std::vector<Decl *> ParsedDecls = Action->takeTopLevelDecls();
  // AST traversals should exclude the preamble, to avoid performance cliffs.
  Clang->getASTContext().setTraversalScope(ParsedDecls);
  if (!CTChecks.empty()) {
    // Run the AST-dependent part of the clang-tidy checks.
    // (The preprocessor part ran already, via PPCallbacks).
    trace::Span Tracer("ClangTidyMatch");
    CTFinder.matchAST(Clang->getASTContext());
  }

  // XXX: This is messy: clang-tidy checks flush some diagnostics at EOF.
  // However Action->EndSourceFile() would destroy the ASTContext!
  // So just inform the preprocessor of EOF, while keeping everything alive.
  Clang->getPreprocessor().EndSourceFile();
  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new IgnoreDiagnostics);
  // CompilerInstance won't run this callback, do it directly.
  ASTDiags.EndSourceFile();

  std::vector<Diag> Diags = CompilerInvocationDiags;
  // Add diagnostics from the preamble, if any.
  if (Preamble)
    Diags.insert(Diags.end(), Preamble->Diags.begin(), Preamble->Diags.end());
  // Finally, add diagnostics coming from the AST.
  {
    std::vector<Diag> D = ASTDiags.take(CTContext.getPointer());
    Diags.insert(Diags.end(), D.begin(), D.end());
  }
  return ParsedAST(Inputs.Version, std::move(Preamble), std::move(Clang),
                   std::move(Action), std::move(Tokens), std::move(Macros),
                   std::move(ParsedDecls), std::move(Diags),
                   std::move(Includes), std::move(CanonIncludes));
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

const MainFileMacros &ParsedAST::getMacros() const { return Macros; }

const std::vector<Diag> &ParsedAST::getDiagnostics() const { return Diags; }

std::size_t ParsedAST::getUsedBytes() const {
  auto &AST = getASTContext();
  // FIXME(ibiryukov): we do not account for the dynamically allocated part of
  // Message and Fixes inside each diagnostic.
  std::size_t Total =
      clangd::getUsedBytes(LocalTopLevelDecls) + clangd::getUsedBytes(Diags);

  // FIXME: the rest of the function is almost a direct copy-paste from
  // libclang's clang_getCXTUResourceUsage. We could share the implementation.

  // Sum up various allocators inside the ast context and the preprocessor.
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

const CanonicalIncludes &ParsedAST::getCanonicalIncludes() const {
  return CanonIncludes;
}

ParsedAST::ParsedAST(llvm::StringRef Version,
                     std::shared_ptr<const PreambleData> Preamble,
                     std::unique_ptr<CompilerInstance> Clang,
                     std::unique_ptr<FrontendAction> Action,
                     syntax::TokenBuffer Tokens, MainFileMacros Macros,
                     std::vector<Decl *> LocalTopLevelDecls,
                     std::vector<Diag> Diags, IncludeStructure Includes,
                     CanonicalIncludes CanonIncludes)
    : Version(Version), Preamble(std::move(Preamble)), Clang(std::move(Clang)),
      Action(std::move(Action)), Tokens(std::move(Tokens)),
      Macros(std::move(Macros)), Diags(std::move(Diags)),
      LocalTopLevelDecls(std::move(LocalTopLevelDecls)),
      Includes(std::move(Includes)), CanonIncludes(std::move(CanonIncludes)) {
  assert(this->Clang);
  assert(this->Action);
}

llvm::Optional<llvm::StringRef> ParsedAST::preambleVersion() const {
  if (!Preamble)
    return llvm::None;
  return llvm::StringRef(Preamble->Version);
}
} // namespace clangd
} // namespace clang
