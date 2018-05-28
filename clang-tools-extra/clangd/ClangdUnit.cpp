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
  DeclTrackingASTConsumer(std::vector<const Decl *> &TopLevelDecls)
      : TopLevelDecls(TopLevelDecls) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const Decl *D : DG) {
      // ObjCMethodDecl are not actually top-level decls.
      if (isa<ObjCMethodDecl>(D))
        continue;

      TopLevelDecls.push_back(D);
    }
    return true;
  }

private:
  std::vector<const Decl *> &TopLevelDecls;
};

class ClangdFrontendAction : public SyntaxOnlyAction {
public:
  std::vector<const Decl *> takeTopLevelDecls() {
    return std::move(TopLevelDecls);
  }

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<DeclTrackingASTConsumer>(/*ref*/ TopLevelDecls);
  }

private:
  std::vector<const Decl *> TopLevelDecls;
};

class CppFilePreambleCallbacks : public PreambleCallbacks {
public:
  CppFilePreambleCallbacks(PathRef File, PreambleParsedCallback ParsedCallback)
      : File(File), ParsedCallback(ParsedCallback) {}

  std::vector<Inclusion> takeInclusions() { return std::move(Inclusions); }

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
    return collectInclusionsInMainFileCallback(
        *SourceMgr,
        [this](Inclusion Inc) { Inclusions.push_back(std::move(Inc)); });
  }

private:
  PathRef File;
  PreambleParsedCallback ParsedCallback;
  std::vector<Inclusion> Inclusions;
  SourceManager *SourceMgr = nullptr;
};

} // namespace

void clangd::dumpAST(ParsedAST &AST, llvm::raw_ostream &OS) {
  AST.getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

llvm::Optional<ParsedAST>
ParsedAST::Build(std::unique_ptr<clang::CompilerInvocation> CI,
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
    log("BeginSourceFile() failed when building AST for " +
        MainInput.getFile());
    return llvm::None;
  }

  std::vector<Inclusion> Inclusions;
  // Copy over the includes from the preamble, then combine with the
  // non-preamble includes below.
  if (Preamble)
    Inclusions = Preamble->Inclusions;

  Clang->getPreprocessor().addPPCallbacks(collectInclusionsInMainFileCallback(
      Clang->getSourceManager(),
      [&Inclusions](Inclusion Inc) { Inclusions.push_back(std::move(Inc)); }));

  if (!Action->Execute())
    log("Execute() failed when building AST for " + MainInput.getFile());

  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new IgnoreDiagnostics);
  // CompilerInstance won't run this callback, do it directly.
  ASTDiags.EndSourceFile();

  std::vector<const Decl *> ParsedDecls = Action->takeTopLevelDecls();
  return ParsedAST(std::move(Preamble), std::move(Clang), std::move(Action),
                   std::move(ParsedDecls), ASTDiags.take(),
                   std::move(Inclusions));
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

ArrayRef<const Decl *> ParsedAST::getLocalTopLevelDecls() {
  return LocalTopLevelDecls;
}

const std::vector<Diag> &ParsedAST::getDiagnostics() const { return Diags; }

std::size_t ParsedAST::getUsedBytes() const {
  auto &AST = getASTContext();
  // FIXME(ibiryukov): we do not account for the dynamically allocated part of
  // Message and Fixes inside each diagnostic.
  return AST.getASTAllocatedMemory() + AST.getSideTableAllocatedMemory() +
         ::getUsedBytes(LocalTopLevelDecls) + ::getUsedBytes(Diags);
}

const std::vector<Inclusion> &ParsedAST::getInclusions() const {
  return Inclusions;
}

PreambleData::PreambleData(PrecompiledPreamble Preamble,
                           std::vector<Diag> Diags,
                           std::vector<Inclusion> Inclusions)
    : Preamble(std::move(Preamble)), Diags(std::move(Diags)),
      Inclusions(std::move(Inclusions)) {}

ParsedAST::ParsedAST(std::shared_ptr<const PreambleData> Preamble,
                     std::unique_ptr<CompilerInstance> Clang,
                     std::unique_ptr<FrontendAction> Action,
                     std::vector<const Decl *> LocalTopLevelDecls,
                     std::vector<Diag> Diags, std::vector<Inclusion> Inclusions)
    : Preamble(std::move(Preamble)), Clang(std::move(Clang)),
      Action(std::move(Action)), Diags(std::move(Diags)),
      LocalTopLevelDecls(std::move(LocalTopLevelDecls)),
      Inclusions(std::move(Inclusions)) {
  assert(this->Clang);
  assert(this->Action);
}

CppFile::CppFile(PathRef FileName, bool StorePreamblesInMemory,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 PreambleParsedCallback PreambleCallback)
    : FileName(FileName), StorePreamblesInMemory(StorePreamblesInMemory),
      PCHs(std::move(PCHs)), PreambleCallback(std::move(PreambleCallback)) {
  log("Created CppFile for " + FileName);
}

llvm::Optional<std::vector<Diag>> CppFile::rebuild(ParseInputs &&Inputs) {
  log("Rebuilding file " + FileName + " with command [" +
      Inputs.CompileCommand.Directory + "] " +
      llvm::join(Inputs.CompileCommand.CommandLine, " "));

  std::vector<const char *> ArgStrs;
  for (const auto &S : Inputs.CompileCommand.CommandLine)
    ArgStrs.push_back(S.c_str());

  if (Inputs.FS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory)) {
    log("Couldn't set working directory");
    // We run parsing anyway, our lit-tests rely on results for non-existing
    // working dirs.
  }

  // Prepare CompilerInvocation.
  std::unique_ptr<CompilerInvocation> CI;
  {
    // FIXME(ibiryukov): store diagnostics from CommandLine when we start
    // reporting them.
    IgnoreDiagnostics IgnoreDiagnostics;
    IntrusiveRefCntPtr<DiagnosticsEngine> CommandLineDiagsEngine =
        CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                            &IgnoreDiagnostics, false);
    CI = createInvocationFromCommandLine(ArgStrs, CommandLineDiagsEngine,
                                         Inputs.FS);
    if (!CI) {
      log("Could not build CompilerInvocation for file " + FileName);
      AST = llvm::None;
      Preamble = nullptr;
      return llvm::None;
    }
    // createInvocationFromCommandLine sets DisableFree.
    CI->getFrontendOpts().DisableFree = false;
    CI->getLangOpts()->CommentOpts.ParseAllComments = true;
  }

  std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(Inputs.Contents, FileName);

  // Compute updated Preamble.
  std::shared_ptr<const PreambleData> NewPreamble =
      rebuildPreamble(*CI, Inputs.CompileCommand, Inputs.FS, *ContentsBuffer);

  // Remove current AST to avoid wasting memory.
  AST = llvm::None;
  // Compute updated AST.
  llvm::Optional<ParsedAST> NewAST;
  {
    trace::Span Tracer("Build");
    SPAN_ATTACH(Tracer, "File", FileName);
    NewAST = ParsedAST::Build(std::move(CI), NewPreamble,
                              std::move(ContentsBuffer), PCHs, Inputs.FS);
  }

  std::vector<Diag> Diagnostics;
  if (NewAST) {
    // Collect diagnostics from both the preamble and the AST.
    if (NewPreamble)
      Diagnostics = NewPreamble->Diags;
    Diagnostics.insert(Diagnostics.end(), NewAST->getDiagnostics().begin(),
                       NewAST->getDiagnostics().end());
  }

  // Write the results of rebuild into class fields.
  Command = std::move(Inputs.CompileCommand);
  Preamble = std::move(NewPreamble);
  AST = std::move(NewAST);
  return Diagnostics;
}

const std::shared_ptr<const PreambleData> &CppFile::getPreamble() const {
  return Preamble;
}

ParsedAST *CppFile::getAST() const {
  // We could add mutable to AST instead of const_cast here, but that would also
  // allow writing to AST from const methods.
  return AST ? const_cast<ParsedAST *>(AST.getPointer()) : nullptr;
}

std::size_t CppFile::getUsedBytes() const {
  std::size_t Total = 0;
  if (AST)
    Total += AST->getUsedBytes();
  if (StorePreamblesInMemory && Preamble)
    Total += Preamble->Preamble.getSize();
  return Total;
}

std::shared_ptr<const PreambleData>
CppFile::rebuildPreamble(CompilerInvocation &CI,
                         const tooling::CompileCommand &Command,
                         IntrusiveRefCntPtr<vfs::FileSystem> FS,
                         llvm::MemoryBuffer &ContentsBuffer) const {
  const auto &OldPreamble = this->Preamble;
  auto Bounds = ComputePreambleBounds(*CI.getLangOpts(), &ContentsBuffer, 0);
  if (OldPreamble && compileCommandsAreEqual(this->Command, Command) &&
      OldPreamble->Preamble.CanReuse(CI, &ContentsBuffer, Bounds, FS.get())) {
    log("Reusing preamble for file " + Twine(FileName));
    return OldPreamble;
  }
  log("Preamble for file " + Twine(FileName) +
      " cannot be reused. Attempting to rebuild it.");

  trace::Span Tracer("Preamble");
  SPAN_ATTACH(Tracer, "File", FileName);
  StoreDiags PreambleDiagnostics;
  IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
      CompilerInstance::createDiagnostics(&CI.getDiagnosticOpts(),
                                          &PreambleDiagnostics, false);

  // Skip function bodies when building the preamble to speed up building
  // the preamble and make it smaller.
  assert(!CI.getFrontendOpts().SkipFunctionBodies);
  CI.getFrontendOpts().SkipFunctionBodies = true;

  CppFilePreambleCallbacks SerializedDeclsCollector(FileName, PreambleCallback);
  auto BuiltPreamble = PrecompiledPreamble::Build(
      CI, &ContentsBuffer, Bounds, *PreambleDiagsEngine, FS, PCHs,
      /*StoreInMemory=*/StorePreamblesInMemory, SerializedDeclsCollector);

  // When building the AST for the main file, we do want the function
  // bodies.
  CI.getFrontendOpts().SkipFunctionBodies = false;

  if (BuiltPreamble) {
    log("Built preamble of size " + Twine(BuiltPreamble->getSize()) +
        " for file " + Twine(FileName));

    return std::make_shared<PreambleData>(
        std::move(*BuiltPreamble), PreambleDiagnostics.take(),
        SerializedDeclsCollector.takeInclusions());
  } else {
    log("Could not build a preamble for file " + Twine(FileName));
    return nullptr;
  }
}

SourceLocation clangd::getBeginningOfIdentifier(ParsedAST &Unit,
                                                const Position &Pos,
                                                const FileID FID) {
  const ASTContext &AST = Unit.getASTContext();
  const SourceManager &SourceMgr = AST.getSourceManager();
  auto Offset = positionToOffset(SourceMgr.getBufferData(FID), Pos);
  if (!Offset) {
    log("getBeginningOfIdentifier: " + toString(Offset.takeError()));
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
