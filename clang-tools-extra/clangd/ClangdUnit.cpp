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
#include "Logger.h"
#include "Trace.h"
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
#include "llvm/Support/Format.h"
#include <algorithm>
#include <chrono>

using namespace clang::clangd;
using namespace clang;

namespace {

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
  std::vector<serialization::DeclID> takeTopLevelDeclIDs() {
    return std::move(TopLevelDeclIDs);
  }

  void AfterPCHEmitted(ASTWriter &Writer) override {
    TopLevelDeclIDs.reserve(TopLevelDecls.size());
    for (Decl *D : TopLevelDecls) {
      // Invalid top-level decls may not have been serialized.
      if (D->isInvalidDecl())
        continue;
      TopLevelDeclIDs.push_back(Writer.getDeclID(D));
    }
  }

  void HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
      if (isa<ObjCMethodDecl>(D))
        continue;
      TopLevelDecls.push_back(D);
    }
  }

private:
  std::vector<Decl *> TopLevelDecls;
  std::vector<serialization::DeclID> TopLevelDeclIDs;
};

/// Convert from clang diagnostic level to LSP severity.
static int getSeverity(DiagnosticsEngine::Level L) {
  switch (L) {
  case DiagnosticsEngine::Remark:
    return 4;
  case DiagnosticsEngine::Note:
    return 3;
  case DiagnosticsEngine::Warning:
    return 2;
  case DiagnosticsEngine::Fatal:
  case DiagnosticsEngine::Error:
    return 1;
  case DiagnosticsEngine::Ignored:
    return 0;
  }
  llvm_unreachable("Unknown diagnostic level!");
}

// Checks whether a location is within a half-open range.
// Note that clang also uses closed source ranges, which this can't handle!
bool locationInRange(SourceLocation L, CharSourceRange R,
                     const SourceManager &M) {
  assert(R.isCharRange());
  if (!R.isValid() || M.getFileID(R.getBegin()) != M.getFileID(R.getEnd()) ||
      M.getFileID(R.getBegin()) != M.getFileID(L))
    return false;
  return L != R.getEnd() && M.isPointWithin(L, R.getBegin(), R.getEnd());
}

// Converts a half-open clang source range to an LSP range.
// Note that clang also uses closed source ranges, which this can't handle!
Range toRange(CharSourceRange R, const SourceManager &M) {
  // Clang is 1-based, LSP uses 0-based indexes.
  return {{static_cast<int>(M.getSpellingLineNumber(R.getBegin())) - 1,
           static_cast<int>(M.getSpellingColumnNumber(R.getBegin())) - 1},
          {static_cast<int>(M.getSpellingLineNumber(R.getEnd())) - 1,
           static_cast<int>(M.getSpellingColumnNumber(R.getEnd())) - 1}};
}

// Clang diags have a location (shown as ^) and 0 or more ranges (~~~~).
// LSP needs a single range.
Range diagnosticRange(const clang::Diagnostic &D, const LangOptions &L) {
  auto &M = D.getSourceManager();
  auto Loc = M.getFileLoc(D.getLocation());
  // Accept the first range that contains the location.
  for (const auto &CR : D.getRanges()) {
    auto R = Lexer::makeFileCharRange(CR, M, L);
    if (locationInRange(Loc, R, M))
      return toRange(R, M);
  }
  // The range may be given as a fixit hint instead.
  for (const auto &F : D.getFixItHints()) {
    auto R = Lexer::makeFileCharRange(F.RemoveRange, M, L);
    if (locationInRange(Loc, R, M))
      return toRange(R, M);
  }
  // If no suitable range is found, just use the token at the location.
  auto R = Lexer::makeFileCharRange(CharSourceRange::getTokenRange(Loc), M, L);
  if (!R.isValid()) // Fall back to location only, let the editor deal with it.
    R = CharSourceRange::getCharRange(Loc);
  return toRange(R, M);
}

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L) {
  TextEdit Result;
  Result.range = toRange(Lexer::makeFileCharRange(FixIt.RemoveRange, M, L), M);
  Result.newText = FixIt.CodeToInsert;
  return Result;
}

llvm::Optional<DiagWithFixIts> toClangdDiag(const clang::Diagnostic &D,
                                            DiagnosticsEngine::Level Level,
                                            const LangOptions &LangOpts) {
  if (!D.hasSourceManager() || !D.getLocation().isValid() ||
      !D.getSourceManager().isInMainFile(D.getLocation()))
    return llvm::None;

  DiagWithFixIts Result;
  Result.Diag.range = diagnosticRange(D, LangOpts);
  Result.Diag.severity = getSeverity(Level);
  SmallString<64> Message;
  D.FormatDiagnostic(Message);
  Result.Diag.message = Message.str();
  for (const FixItHint &Fix : D.getFixItHints())
    Result.FixIts.push_back(toTextEdit(Fix, D.getSourceManager(), LangOpts));
  return std::move(Result);
}

class StoreDiagsConsumer : public DiagnosticConsumer {
public:
  StoreDiagsConsumer(std::vector<DiagWithFixIts> &Output) : Output(Output) {}

  // Track language options in case we need to expand token ranges.
  void BeginSourceFile(const LangOptions &Opts, const Preprocessor *) override {
    LangOpts = Opts;
  }

  void EndSourceFile() override { LangOpts = llvm::None; }

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {
    DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);

    if (LangOpts)
      if (auto D = toClangdDiag(Info, DiagLevel, *LangOpts))
        Output.push_back(std::move(*D));
  }

private:
  std::vector<DiagWithFixIts> &Output;
  llvm::Optional<LangOptions> LangOpts;
};

template <class T> bool futureIsReady(std::shared_future<T> const &Future) {
  return Future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

} // namespace

void clangd::dumpAST(ParsedAST &AST, llvm::raw_ostream &OS) {
  AST.getASTContext().getTranslationUnitDecl()->dump(OS, true);
}

llvm::Optional<ParsedAST>
ParsedAST::Build(const Context &Ctx,
                 std::unique_ptr<clang::CompilerInvocation> CI,
                 std::shared_ptr<const PreambleData> Preamble,
                 std::unique_ptr<llvm::MemoryBuffer> Buffer,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 IntrusiveRefCntPtr<vfs::FileSystem> VFS) {

  std::vector<DiagWithFixIts> ASTDiags;
  StoreDiagsConsumer UnitDiagsConsumer(/*ref*/ ASTDiags);

  const PrecompiledPreamble *PreamblePCH =
      Preamble ? &Preamble->Preamble : nullptr;
  auto Clang = prepareCompilerInstance(
      std::move(CI), PreamblePCH, std::move(Buffer), std::move(PCHs),
      std::move(VFS), /*ref*/ UnitDiagsConsumer);

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInstance> CICleanup(
      Clang.get());

  auto Action = llvm::make_unique<ClangdFrontendAction>();
  const FrontendInputFile &MainInput = Clang->getFrontendOpts().Inputs[0];
  if (!Action->BeginSourceFile(*Clang, MainInput)) {
    log(Ctx, "BeginSourceFile() failed when building AST for " +
                 MainInput.getFile());
    return llvm::None;
  }
  if (!Action->Execute())
    log(Ctx, "Execute() failed when building AST for " + MainInput.getFile());

  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new IgnoreDiagnostics);

  std::vector<const Decl *> ParsedDecls = Action->takeTopLevelDecls();
  return ParsedAST(std::move(Preamble), std::move(Clang), std::move(Action),
                   std::move(ParsedDecls), std::move(ASTDiags));
}

namespace {

SourceLocation getMacroArgExpandedLocation(const SourceManager &Mgr,
                                           const FileEntry *FE, Position Pos) {
  SourceLocation InputLoc =
      Mgr.translateFileLineCol(FE, Pos.line + 1, Pos.character + 1);
  return Mgr.getMacroArgExpandedLocation(InputLoc);
}

} // namespace

void ParsedAST::ensurePreambleDeclsDeserialized() {
  if (PreambleDeclsDeserialized || !Preamble)
    return;

  std::vector<const Decl *> Resolved;
  Resolved.reserve(Preamble->TopLevelDeclIDs.size());

  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (serialization::DeclID TopLevelDecl : Preamble->TopLevelDeclIDs) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    if (Decl *D = Source.GetExternalDecl(TopLevelDecl))
      Resolved.push_back(D);
  }

  TopLevelDecls.reserve(TopLevelDecls.size() +
                        Preamble->TopLevelDeclIDs.size());
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());

  PreambleDeclsDeserialized = true;
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

ArrayRef<const Decl *> ParsedAST::getTopLevelDecls() {
  ensurePreambleDeclsDeserialized();
  return TopLevelDecls;
}

const std::vector<DiagWithFixIts> &ParsedAST::getDiagnostics() const {
  return Diags;
}

PreambleData::PreambleData(PrecompiledPreamble Preamble,
                           std::vector<serialization::DeclID> TopLevelDeclIDs,
                           std::vector<DiagWithFixIts> Diags)
    : Preamble(std::move(Preamble)),
      TopLevelDeclIDs(std::move(TopLevelDeclIDs)), Diags(std::move(Diags)) {}

ParsedAST::ParsedAST(std::shared_ptr<const PreambleData> Preamble,
                     std::unique_ptr<CompilerInstance> Clang,
                     std::unique_ptr<FrontendAction> Action,
                     std::vector<const Decl *> TopLevelDecls,
                     std::vector<DiagWithFixIts> Diags)
    : Preamble(std::move(Preamble)), Clang(std::move(Clang)),
      Action(std::move(Action)), Diags(std::move(Diags)),
      TopLevelDecls(std::move(TopLevelDecls)),
      PreambleDeclsDeserialized(false) {
  assert(this->Clang);
  assert(this->Action);
}

ParsedASTWrapper::ParsedASTWrapper(ParsedASTWrapper &&Wrapper)
    : AST(std::move(Wrapper.AST)) {}

ParsedASTWrapper::ParsedASTWrapper(llvm::Optional<ParsedAST> AST)
    : AST(std::move(AST)) {}

std::shared_ptr<CppFile>
CppFile::Create(PathRef FileName, bool StorePreamblesInMemory,
                std::shared_ptr<PCHContainerOperations> PCHs,
                ASTParsedCallback ASTCallback) {
  return std::shared_ptr<CppFile>(new CppFile(FileName, StorePreamblesInMemory,
                                              std::move(PCHs),
                                              std::move(ASTCallback)));
}

CppFile::CppFile(PathRef FileName, bool StorePreamblesInMemory,
                 std::shared_ptr<PCHContainerOperations> PCHs,
                 ASTParsedCallback ASTCallback)
    : FileName(FileName), StorePreamblesInMemory(StorePreamblesInMemory),
      RebuildCounter(0), RebuildInProgress(false), PCHs(std::move(PCHs)),
      ASTCallback(std::move(ASTCallback)) {
  // FIXME(ibiryukov): we should pass a proper Context here.
  log(Context::empty(), "Created CppFile for " + FileName);

  std::lock_guard<std::mutex> Lock(Mutex);
  LatestAvailablePreamble = nullptr;
  PreamblePromise.set_value(nullptr);
  PreambleFuture = PreamblePromise.get_future();

  ASTPromise.set_value(std::make_shared<ParsedASTWrapper>(llvm::None));
  ASTFuture = ASTPromise.get_future();
}

void CppFile::cancelRebuild() { deferCancelRebuild()(); }

UniqueFunction<void()> CppFile::deferCancelRebuild() {
  std::unique_lock<std::mutex> Lock(Mutex);
  // Cancel an ongoing rebuild, if any, and wait for it to finish.
  unsigned RequestRebuildCounter = ++this->RebuildCounter;
  // Rebuild asserts that futures aren't ready if rebuild is cancelled.
  // We want to keep this invariant.
  if (futureIsReady(PreambleFuture)) {
    PreamblePromise = std::promise<std::shared_ptr<const PreambleData>>();
    PreambleFuture = PreamblePromise.get_future();
  }
  if (futureIsReady(ASTFuture)) {
    ASTPromise = std::promise<std::shared_ptr<ParsedASTWrapper>>();
    ASTFuture = ASTPromise.get_future();
  }

  Lock.unlock();
  // Notify about changes to RebuildCounter.
  RebuildCond.notify_all();

  std::shared_ptr<CppFile> That = shared_from_this();
  return [That, RequestRebuildCounter]() {
    std::unique_lock<std::mutex> Lock(That->Mutex);
    CppFile *This = &*That;
    This->RebuildCond.wait(Lock, [This, RequestRebuildCounter]() {
      return !This->RebuildInProgress ||
             This->RebuildCounter != RequestRebuildCounter;
    });

    // This computation got cancelled itself, do nothing.
    if (This->RebuildCounter != RequestRebuildCounter)
      return;

    // Set empty results for Promises.
    That->PreamblePromise.set_value(nullptr);
    That->ASTPromise.set_value(std::make_shared<ParsedASTWrapper>(llvm::None));
  };
}

llvm::Optional<std::vector<DiagWithFixIts>>
CppFile::rebuild(const Context &Ctx, ParseInputs &&Inputs) {
  return deferRebuild(std::move(Inputs))(Ctx);
}

UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>(const Context &)>
CppFile::deferRebuild(ParseInputs &&Inputs) {
  std::shared_ptr<const PreambleData> OldPreamble;
  std::shared_ptr<PCHContainerOperations> PCHs;
  unsigned RequestRebuildCounter;
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    // Increase RebuildCounter to cancel all ongoing FinishRebuild operations.
    // They will try to exit as early as possible and won't call set_value on
    // our promises.
    RequestRebuildCounter = ++this->RebuildCounter;
    PCHs = this->PCHs;

    // Remember the preamble to be used during rebuild.
    OldPreamble = this->LatestAvailablePreamble;
    // Setup std::promises and std::futures for Preamble and AST. Corresponding
    // futures will wait until the rebuild process is finished.
    if (futureIsReady(this->PreambleFuture)) {
      this->PreamblePromise =
          std::promise<std::shared_ptr<const PreambleData>>();
      this->PreambleFuture = this->PreamblePromise.get_future();
    }
    if (futureIsReady(this->ASTFuture)) {
      this->ASTPromise = std::promise<std::shared_ptr<ParsedASTWrapper>>();
      this->ASTFuture = this->ASTPromise.get_future();
    }
    this->LastCommand = Inputs.CompileCommand;
  } // unlock Mutex.
  // Notify about changes to RebuildCounter.
  RebuildCond.notify_all();

  // A helper to function to finish the rebuild. May be run on a different
  // thread.

  // Don't let this CppFile die before rebuild is finished.
  std::shared_ptr<CppFile> That = shared_from_this();
  auto FinishRebuild =
      [OldPreamble, RequestRebuildCounter, PCHs,
       That](ParseInputs Inputs,
             const Context &Ctx) mutable /* to allow changing OldPreamble. */
      -> llvm::Optional<std::vector<DiagWithFixIts>> {
    log(Context::empty(),
        "Rebuilding file " + That->FileName + " with command [" +
            Inputs.CompileCommand.Directory + "] " +
            llvm::join(Inputs.CompileCommand.CommandLine, " "));

    // Only one execution of this method is possible at a time.
    // RebuildGuard will wait for any ongoing rebuilds to finish and will put us
    // into a state for doing a rebuild.
    RebuildGuard Rebuild(*That, RequestRebuildCounter);
    if (Rebuild.wasCancelledBeforeConstruction())
      return llvm::None;

    std::vector<const char *> ArgStrs;
    for (const auto &S : Inputs.CompileCommand.CommandLine)
      ArgStrs.push_back(S.c_str());

    Inputs.FS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory);

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
      // createInvocationFromCommandLine sets DisableFree.
      CI->getFrontendOpts().DisableFree = false;
    }
    assert(CI && "Couldn't create CompilerInvocation");

    std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
        llvm::MemoryBuffer::getMemBufferCopy(Inputs.Contents, That->FileName);

    // A helper function to rebuild the preamble or reuse the existing one. Does
    // not mutate any fields of CppFile, only does the actual computation.
    // Lamdba is marked mutable to call reset() on OldPreamble.
    auto DoRebuildPreamble =
        [&]() mutable -> std::shared_ptr<const PreambleData> {
      auto Bounds =
          ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0);
      if (OldPreamble &&
          OldPreamble->Preamble.CanReuse(*CI, ContentsBuffer.get(), Bounds,
                                         Inputs.FS.get())) {
        log(Ctx, "Reusing preamble for file " + Twine(That->FileName));
        return OldPreamble;
      }
      log(Ctx, "Premble for file " + Twine(That->FileName) +
                   " cannot be reused. Attempting to rebuild it.");
      // We won't need the OldPreamble anymore, release it so it can be
      // deleted (if there are no other references to it).
      OldPreamble.reset();

      trace::Span Tracer(Ctx, "Preamble");
      SPAN_ATTACH(Tracer, "File", That->FileName);
      std::vector<DiagWithFixIts> PreambleDiags;
      StoreDiagsConsumer PreambleDiagnosticsConsumer(/*ref*/ PreambleDiags);
      IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
          CompilerInstance::createDiagnostics(
              &CI->getDiagnosticOpts(), &PreambleDiagnosticsConsumer, false);

      // Skip function bodies when building the preamble to speed up building
      // the preamble and make it smaller.
      assert(!CI->getFrontendOpts().SkipFunctionBodies);
      CI->getFrontendOpts().SkipFunctionBodies = true;

      CppFilePreambleCallbacks SerializedDeclsCollector;
      auto BuiltPreamble = PrecompiledPreamble::Build(
          *CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine, Inputs.FS,
          PCHs,
          /*StoreInMemory=*/That->StorePreamblesInMemory,
          SerializedDeclsCollector);

      // When building the AST for the main file, we do want the function
      // bodies.
      CI->getFrontendOpts().SkipFunctionBodies = false;

      if (BuiltPreamble) {
        log(Ctx, "Built preamble of size " + Twine(BuiltPreamble->getSize()) +
                     " for file " + Twine(That->FileName));

        return std::make_shared<PreambleData>(
            std::move(*BuiltPreamble),
            SerializedDeclsCollector.takeTopLevelDeclIDs(),
            std::move(PreambleDiags));
      } else {
        log(Ctx,
            "Could not build a preamble for file " + Twine(That->FileName));
        return nullptr;
      }
    };

    // Compute updated Preamble.
    std::shared_ptr<const PreambleData> NewPreamble = DoRebuildPreamble();
    // Publish the new Preamble.
    {
      std::lock_guard<std::mutex> Lock(That->Mutex);
      // We always set LatestAvailablePreamble to the new value, hoping that it
      // will still be usable in the further requests.
      That->LatestAvailablePreamble = NewPreamble;
      if (RequestRebuildCounter != That->RebuildCounter)
        return llvm::None; // Our rebuild request was cancelled, do nothing.
      That->PreamblePromise.set_value(NewPreamble);
    } // unlock Mutex

    // Prepare the Preamble and supplementary data for rebuilding AST.
    std::vector<DiagWithFixIts> Diagnostics;
    if (NewPreamble) {
      Diagnostics.insert(Diagnostics.begin(), NewPreamble->Diags.begin(),
                         NewPreamble->Diags.end());
    }

    // Compute updated AST.
    llvm::Optional<ParsedAST> NewAST;
    {
      trace::Span Tracer(Ctx, "Build");
      SPAN_ATTACH(Tracer, "File", That->FileName);
      NewAST = ParsedAST::Build(Ctx, std::move(CI), std::move(NewPreamble),
                                std::move(ContentsBuffer), PCHs, Inputs.FS);
    }

    if (NewAST) {
      Diagnostics.insert(Diagnostics.end(), NewAST->getDiagnostics().begin(),
                         NewAST->getDiagnostics().end());
      if (That->ASTCallback)
        That->ASTCallback(Ctx, That->FileName, NewAST.getPointer());
    } else {
      // Don't report even Preamble diagnostics if we coulnd't build AST.
      Diagnostics.clear();
    }

    // Publish the new AST.
    {
      std::lock_guard<std::mutex> Lock(That->Mutex);
      if (RequestRebuildCounter != That->RebuildCounter)
        return Diagnostics; // Our rebuild request was cancelled, don't set
                            // ASTPromise.

      That->ASTPromise.set_value(
          std::make_shared<ParsedASTWrapper>(std::move(NewAST)));
    } // unlock Mutex

    return Diagnostics;
  };

  return BindWithForward(FinishRebuild, std::move(Inputs));
}

std::shared_future<std::shared_ptr<const PreambleData>>
CppFile::getPreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return PreambleFuture;
}

std::shared_ptr<const PreambleData> CppFile::getPossiblyStalePreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LatestAvailablePreamble;
}

std::shared_future<std::shared_ptr<ParsedASTWrapper>> CppFile::getAST() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return ASTFuture;
}

llvm::Optional<tooling::CompileCommand> CppFile::getLastCommand() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LastCommand;
}

CppFile::RebuildGuard::RebuildGuard(CppFile &File,
                                    unsigned RequestRebuildCounter)
    : File(File), RequestRebuildCounter(RequestRebuildCounter) {
  std::unique_lock<std::mutex> Lock(File.Mutex);
  WasCancelledBeforeConstruction = File.RebuildCounter != RequestRebuildCounter;
  if (WasCancelledBeforeConstruction)
    return;

  File.RebuildCond.wait(Lock, [&File, RequestRebuildCounter]() {
    return !File.RebuildInProgress ||
           File.RebuildCounter != RequestRebuildCounter;
  });

  WasCancelledBeforeConstruction = File.RebuildCounter != RequestRebuildCounter;
  if (WasCancelledBeforeConstruction)
    return;

  File.RebuildInProgress = true;
}

bool CppFile::RebuildGuard::wasCancelledBeforeConstruction() const {
  return WasCancelledBeforeConstruction;
}

CppFile::RebuildGuard::~RebuildGuard() {
  if (WasCancelledBeforeConstruction)
    return;

  std::unique_lock<std::mutex> Lock(File.Mutex);
  assert(File.RebuildInProgress);
  File.RebuildInProgress = false;

  if (File.RebuildCounter == RequestRebuildCounter) {
    // Our rebuild request was successful.
    assert(futureIsReady(File.ASTFuture));
    assert(futureIsReady(File.PreambleFuture));
  } else {
    // Our rebuild request was cancelled, because further reparse was requested.
    assert(!futureIsReady(File.ASTFuture));
    assert(!futureIsReady(File.PreambleFuture));
  }

  Lock.unlock();
  File.RebuildCond.notify_all();
}

SourceLocation clangd::getBeginningOfIdentifier(ParsedAST &Unit,
                                                const Position &Pos,
                                                const FileEntry *FE) {
  // The language server protocol uses zero-based line and column numbers.
  // Clang uses one-based numbers.

  const ASTContext &AST = Unit.getASTContext();
  const SourceManager &SourceMgr = AST.getSourceManager();

  SourceLocation InputLocation =
      getMacroArgExpandedLocation(SourceMgr, FE, Pos);
  if (Pos.character == 0) {
    return InputLocation;
  }

  // This handle cases where the position is in the middle of a token or right
  // after the end of a token. In theory we could just use GetBeginningOfToken
  // to find the start of the token at the input position, but this doesn't
  // work when right after the end, i.e. foo|.
  // So try to go back by one and see if we're still inside the an identifier
  // token. If so, Take the beginning of this token.
  // (It should be the same identifier because you can't have two adjacent
  // identifiers without another token in between.)
  SourceLocation PeekBeforeLocation = getMacroArgExpandedLocation(
      SourceMgr, FE, Position{Pos.line, Pos.character - 1});
  Token Result;
  if (Lexer::getRawToken(PeekBeforeLocation, Result, SourceMgr,
                         AST.getLangOpts(), false)) {
    // getRawToken failed, just use InputLocation.
    return InputLocation;
  }

  if (Result.is(tok::raw_identifier)) {
    return Lexer::GetBeginningOfToken(PeekBeforeLocation, SourceMgr,
                                      AST.getLangOpts());
  }

  return InputLocation;
}
