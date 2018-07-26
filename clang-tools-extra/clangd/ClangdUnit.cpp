//===--- ClangdUnit.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "ClangdUnit.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace clang::clangd;
using namespace clang;

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
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
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
    ParsedCallback(File, CI.getASTContext(), CI.getPreprocessorPtr());
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

} // namespace

void clangd::dumpAST(ParsedAST &AST, llvm::raw_ostream &OS) {
  AST.getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

llvm::Optional<ParsedAST>
ParsedAST::build(std::unique_ptr<clang::CompilerInvocation> CI,
                 std::shared_ptr<const PreambleData> Preamble,
                 std::unique_ptr<llvm::MemoryBuffer> Buffer,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
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
    return llvm::None;

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance> CICleanup(
      Clang.get());

  auto Action = llvm::make_unique<ClangdFrontendAction>();
  const FrontendInputFile &MainInput = Clang->getFrontendOpts().Inputs[0];
  if (!Action->BeginSourceFile(*Clang, MainInput)) {
    log("BeginSourceFile() failed when building AST for {0}",
        MainInput.getFile());
    return llvm::None;
  }

  // Copy over the includes from the preamble, then combine with the
  // non-preamble includes below.
  auto Includes = Preamble ? Preamble->Includes : IncludeStructure{};
  Clang->getPreprocessor().addPPCallbacks(
      collectIncludeStructureCallback(Clang->getSourceManager(), &Includes));

  if (!Action->Execute())
    log("Execute() failed when building AST for {0}", MainInput.getFile());

  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new IgnoreDiagnostics);
  // CompilerInstance won't run this callback, do it directly.
  ASTDiags.EndSourceFile();

  std::vector<Decl *> ParsedDecls = Action->takeTopLevelDecls();
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
    Action->EndSourceFile();
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

ArrayRef<Decl *> ParsedAST::getLocalTopLevelDecls() {
  return LocalTopLevelDecls;
}

const std::vector<Diag> &ParsedAST::getDiagnostics() const { return Diags; }

std::size_t ParsedAST::getUsedBytes() const {
  auto &AST = getASTContext();
  // FIXME(ibiryukov): we do not account for the dynamically allocated part of
  // Message and Fixes inside each diagnostic.
  std::size_t Total =
      ::getUsedBytes(LocalTopLevelDecls) + ::getUsedBytes(Diags);

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
                           std::vector<Diag> Diags, IncludeStructure Includes)
    : Preamble(std::move(Preamble)), Diags(std::move(Diags)),
      Includes(std::move(Includes)) {}

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

std::unique_ptr<CompilerInvocation>
clangd::buildCompilerInvocation(const ParseInputs &Inputs) {
  std::vector<const char *> ArgStrs;
  for (const auto &S : Inputs.CompileCommand.CommandLine)
    ArgStrs.push_back(S.c_str());

  if (Inputs.FS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory)) {
    log("Couldn't set working directory when creating compiler invocation.");
    // We proceed anyway, our lit-tests rely on results for non-existing working
    // dirs.
  }

  // FIXME(ibiryukov): store diagnostics from CommandLine when we start
  // reporting them.
  IgnoreDiagnostics IgnoreDiagnostics;
  IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
      CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                          &IgnoreDiagnostics, false);
  std::unique_ptr<CompilerInvocation> CI = createInvocationFromCommandLine(
      ArgStrs, CommandLineDiagsEngine, Inputs.FS);
  if (!CI)
    return nullptr;
  // createInvocationFromCommandLine sets DisableFree.
  CI->getFrontendOpts().DisableFree = false;
  CI->getLangOpts()->CommentOpts.ParseAllComments = true;
  return CI;
}

std::shared_ptr<const PreambleData> clangd::buildPreamble(
    PathRef FileName, CompilerInvocation &CI,
    std::shared_ptr<const PreambleData> OldPreamble,
    const tooling::CompileCommand &OldCompileCommand, const ParseInputs &Inputs,
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
    vlog("Reusing preamble for file {0}", Twine(FileName));
    return OldPreamble;
  }
  vlog("Preamble for file {0} cannot be reused. Attempting to rebuild it.",
       FileName);

  trace::Span Tracer("BuildPreamble");
  SPAN_ATTACH(Tracer, "File", FileName);
  StoreDiags PreambleDiagnostics;
  IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
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
  auto BuiltPreamble = PrecompiledPreamble::Build(
      CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine, Inputs.FS, PCHs,
      StoreInMemory, SerializedDeclsCollector);

  // When building the AST for the main file, we do want the function
  // bodies.
  CI.getFrontendOpts().SkipFunctionBodies = false;

  if (BuiltPreamble) {
    vlog("Built preamble of size {0} for file {1}", BuiltPreamble->getSize(),
         FileName);
    return std::make_shared<PreambleData>(
        std::move(*BuiltPreamble), PreambleDiagnostics.take(),
        SerializedDeclsCollector.takeIncludes());
  } else {
    elog("Could not build a preamble for file {0}", FileName);
    return nullptr;
  }
}

llvm::Optional<ParsedAST> clangd::buildAST(
    PathRef FileName, std::unique_ptr<CompilerInvocation> Invocation,
    const ParseInputs &Inputs, std::shared_ptr<const PreambleData> Preamble,
    std::shared_ptr<PCHContainerOperations> PCHs) {
  trace::Span Tracer("BuildAST");
  SPAN_ATTACH(Tracer, "File", FileName);

  if (Inputs.FS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory)) {
    log("Couldn't set working directory when building the preamble.");
    // We proceed anyway, our lit-tests rely on results for non-existing working
    // dirs.
  }

  return ParsedAST::build(
      llvm::make_unique<CompilerInvocation>(*Invocation), Preamble,
      llvm::MemoryBuffer::getMemBufferCopy(Inputs.Contents), PCHs, Inputs.FS);
}

SourceLocation clangd::getBeginningOfIdentifier(ParsedAST &Unit,
                                                const Position &Pos,
                                                const FileID FID) {
  const ASTContext &AST = Unit.getASTContext();
  const SourceManager &SourceMgr = AST.getSourceManager();
  auto Offset = positionToOffset(SourceMgr.getBufferData(FID), Pos);
  if (!Offset) {
    log("getBeginningOfIdentifier: {0}", Offset.takeError());
    return SourceLocation();
  }
  SourceLocation InputLoc = SourceMgr.getComposedLoc(FID, *Offset);

  // GetBeginningOfToken(pos) is almost what we want, but does the wrong thing
  // if the cursor is at the end of the identifier.
  // Instead, we lex at GetBeginningOfToken(pos - 1). The cases are:
  //  1) at the beginning of an identifier, we'll be looking at something
  //  that isn't an identifier.
  //  2) at the middle or end of an identifier, we get the identifier.
  //  3) anywhere outside an identifier, we'll get some non-identifier thing.
  // We can't actually distinguish cases 1 and 3, but returning the original
  // location is correct for both!
  if (*Offset == 0) // Case 1 or 3.
    return SourceMgr.getMacroArgExpandedLocation(InputLoc);
  SourceLocation Before =
      SourceMgr.getMacroArgExpandedLocation(InputLoc.getLocWithOffset(-1));
  Before = Lexer::GetBeginningOfToken(Before, SourceMgr, AST.getLangOpts());
  Token Tok;
  if (Before.isValid() &&
      !Lexer::getRawToken(Before, Tok, SourceMgr, AST.getLangOpts(), false) &&
      Tok.is(tok::raw_identifier))
    return Before;                                        // Case 2.
  return SourceMgr.getMacroArgExpandedLocation(InputLoc); // Case 1 or 3.
}
