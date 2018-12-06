//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "CodeComplete.h"
#include "FindSymbols.h"
#include "Headers.h"
#include "SourceCode.h"
#include "Trace.h"
#include "XRefs.h"
#include "index/FileIndex.h"
#include "index/Merge.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Refactoring/RefactoringResultConsumer.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <future>
#include <mutex>

using namespace llvm;
namespace clang {
namespace clangd {
namespace {

std::string getStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

class RefactoringResultCollector final
    : public tooling::RefactoringResultConsumer {
public:
  void handleError(Error Err) override {
    assert(!Result.hasValue());
    // FIXME: figure out a way to return better message for DiagnosticError.
    // clangd uses llvm::toString to convert the Err to string, however, for
    // DiagnosticError, only "clang diagnostic" will be generated.
    Result = std::move(Err);
  }

  // Using the handle(SymbolOccurrences) from parent class.
  using tooling::RefactoringResultConsumer::handle;

  void handle(tooling::AtomicChanges SourceReplacements) override {
    assert(!Result.hasValue());
    Result = std::move(SourceReplacements);
  }

  Optional<Expected<tooling::AtomicChanges>> Result;
};

// Update the FileIndex with new ASTs and plumb the diagnostics responses.
struct UpdateIndexCallbacks : public ParsingCallbacks {
  UpdateIndexCallbacks(FileIndex *FIndex, DiagnosticsConsumer &DiagConsumer)
      : FIndex(FIndex), DiagConsumer(DiagConsumer) {}

  void onPreambleAST(PathRef Path, ASTContext &Ctx,
                     std::shared_ptr<clang::Preprocessor> PP) override {
    if (FIndex)
      FIndex->updatePreamble(Path, Ctx, std::move(PP));
  }

  void onMainAST(PathRef Path, ParsedAST &AST) override {
    if (FIndex)
      FIndex->updateMain(Path, AST);
  }

  void onDiagnostics(PathRef File, std::vector<Diag> Diags) override {
    DiagConsumer.onDiagnosticsReady(File, std::move(Diags));
  }

  void onFileUpdated(PathRef File, const TUStatus &Status) override {
    DiagConsumer.onFileUpdated(File, Status);
  }

private:
  FileIndex *FIndex;
  DiagnosticsConsumer &DiagConsumer;
};
} // namespace

ClangdServer::Options ClangdServer::optsForTest() {
  ClangdServer::Options Opts;
  Opts.UpdateDebounce = std::chrono::steady_clock::duration::zero(); // Faster!
  Opts.StorePreamblesInMemory = true;
  Opts.AsyncThreadsCount = 4; // Consistent!
  return Opts;
}

ClangdServer::ClangdServer(const GlobalCompilationDatabase &CDB,
                           const FileSystemProvider &FSProvider,
                           DiagnosticsConsumer &DiagConsumer,
                           const Options &Opts)
    : CDB(CDB), FSProvider(FSProvider),
      ResourceDir(Opts.ResourceDir ? *Opts.ResourceDir
                                   : getStandardResourceDir()),
      DynamicIdx(Opts.BuildDynamicSymbolIndex
                     ? new FileIndex(Opts.HeavyweightDynamicSymbolIndex)
                     : nullptr),
      WorkspaceRoot(Opts.WorkspaceRoot),
      PCHs(std::make_shared<PCHContainerOperations>()),
      // Pass a callback into `WorkScheduler` to extract symbols from a newly
      // parsed file and rebuild the file index synchronously each time an AST
      // is parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      WorkScheduler(Opts.AsyncThreadsCount, Opts.StorePreamblesInMemory,
                    llvm::make_unique<UpdateIndexCallbacks>(DynamicIdx.get(),
                                                            DiagConsumer),
                    Opts.UpdateDebounce, Opts.RetentionPolicy) {
  // Adds an index to the stack, at higher priority than existing indexes.
  auto AddIndex = [&](SymbolIndex *Idx) {
    if (this->Index != nullptr) {
      MergedIdx.push_back(llvm::make_unique<MergedIndex>(Idx, this->Index));
      this->Index = MergedIdx.back().get();
    } else {
      this->Index = Idx;
    }
  };
  if (Opts.StaticIndex)
    AddIndex(Opts.StaticIndex);
  if (Opts.BackgroundIndex) {
    BackgroundIdx = llvm::make_unique<BackgroundIndex>(
        Context::current().clone(), ResourceDir, FSProvider, CDB,
        BackgroundIndexStorage::createDiskBackedStorageFactory());
    AddIndex(BackgroundIdx.get());
  }
  if (DynamicIdx)
    AddIndex(DynamicIdx.get());
}

void ClangdServer::addDocument(PathRef File, StringRef Contents,
                               WantDiagnostics WantDiags) {
  // FIXME: some build systems like Bazel will take time to preparing
  // environment to build the file, it would be nice if we could emit a
  // "PreparingBuild" status to inform users, it is non-trivial given the
  // current implementation.
  WorkScheduler.update(File,
                       ParseInputs{getCompileCommand(File),
                                   FSProvider.getFileSystem(), Contents.str()},
                       WantDiags);
}

void ClangdServer::removeDocument(PathRef File) {
  WorkScheduler.remove(File);
}

void ClangdServer::codeComplete(PathRef File, Position Pos,
                                const clangd::CodeCompleteOptions &Opts,
                                Callback<CodeCompleteResult> CB) {
  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  // Copy PCHs to avoid accessing this->PCHs concurrently
  std::shared_ptr<PCHContainerOperations> PCHs = this->PCHs;
  auto FS = FSProvider.getFileSystem();

  auto Task = [PCHs, Pos, FS, CodeCompleteOpts,
               this](Path File, Callback<CodeCompleteResult> CB,
                     Expected<InputsAndPreamble> IP) {
    if (!IP)
      return CB(IP.takeError());
    if (isCancelled())
      return CB(make_error<CancelledError>());

    Optional<SpeculativeFuzzyFind> SpecFuzzyFind;
    if (CodeCompleteOpts.Index && CodeCompleteOpts.SpeculativeIndexRequest) {
      SpecFuzzyFind.emplace();
      {
        std::lock_guard<std::mutex> Lock(CachedCompletionFuzzyFindRequestMutex);
        SpecFuzzyFind->CachedReq = CachedCompletionFuzzyFindRequestByFile[File];
      }
    }

    // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
    // both the old and the new version in case only one of them matches.
    CodeCompleteResult Result = clangd::codeComplete(
        File, IP->Command, IP->Preamble, IP->Contents, Pos, FS, PCHs,
        CodeCompleteOpts, SpecFuzzyFind ? SpecFuzzyFind.getPointer() : nullptr);
    {
      clang::clangd::trace::Span Tracer("Completion results callback");
      CB(std::move(Result));
    }
    if (SpecFuzzyFind && SpecFuzzyFind->NewReq.hasValue()) {
      std::lock_guard<std::mutex> Lock(CachedCompletionFuzzyFindRequestMutex);
      CachedCompletionFuzzyFindRequestByFile[File] =
          SpecFuzzyFind->NewReq.getValue();
    }
    // SpecFuzzyFind is only destroyed after speculative fuzzy find finishes.
    // We don't want `codeComplete` to wait for the async call if it doesn't use
    // the result (e.g. non-index completion, speculation fails), so that `CB`
    // is called as soon as results are available.
  };

  // We use a potentially-stale preamble because latency is critical here.
  WorkScheduler.runWithPreamble("CodeComplete", File, TUScheduler::Stale,
                                Bind(Task, File.str(), std::move(CB)));
}

void ClangdServer::signatureHelp(PathRef File, Position Pos,
                                 Callback<SignatureHelp> CB) {

  auto PCHs = this->PCHs;
  auto FS = FSProvider.getFileSystem();
  auto *Index = this->Index;
  auto Action = [Pos, FS, PCHs, Index](Path File, Callback<SignatureHelp> CB,
                                       Expected<InputsAndPreamble> IP) {
    if (!IP)
      return CB(IP.takeError());

    auto PreambleData = IP->Preamble;
    CB(clangd::signatureHelp(File, IP->Command, PreambleData, IP->Contents, Pos,
                             FS, PCHs, Index));
  };

  // Unlike code completion, we wait for an up-to-date preamble here.
  // Signature help is often triggered after code completion. If the code
  // completion inserted a header to make the symbol available, then using
  // the old preamble would yield useless results.
  WorkScheduler.runWithPreamble("SignatureHelp", File, TUScheduler::Consistent,
                                Bind(Action, File.str(), std::move(CB)));
}

Expected<tooling::Replacements>
ClangdServer::formatRange(StringRef Code, PathRef File, Range Rng) {
  Expected<size_t> Begin = positionToOffset(Code, Rng.start);
  if (!Begin)
    return Begin.takeError();
  Expected<size_t> End = positionToOffset(Code, Rng.end);
  if (!End)
    return End.takeError();
  return formatCode(Code, File, {tooling::Range(*Begin, *End - *Begin)});
}

Expected<tooling::Replacements> ClangdServer::formatFile(StringRef Code,
                                                         PathRef File) {
  // Format everything.
  return formatCode(Code, File, {tooling::Range(0, Code.size())});
}

Expected<tooling::Replacements>
ClangdServer::formatOnType(StringRef Code, PathRef File, Position Pos) {
  // Look for the previous opening brace from the character position and
  // format starting from there.
  Expected<size_t> CursorPos = positionToOffset(Code, Pos);
  if (!CursorPos)
    return CursorPos.takeError();
  size_t PreviousLBracePos = StringRef(Code).find_last_of('{', *CursorPos);
  if (PreviousLBracePos == StringRef::npos)
    PreviousLBracePos = *CursorPos;
  size_t Len = *CursorPos - PreviousLBracePos;

  return formatCode(Code, File, {tooling::Range(PreviousLBracePos, Len)});
}

void ClangdServer::rename(PathRef File, Position Pos, StringRef NewName,
                          Callback<std::vector<tooling::Replacement>> CB) {
  auto Action = [Pos](Path File, std::string NewName,
                      Callback<std::vector<tooling::Replacement>> CB,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto &AST = InpAST->AST;

    RefactoringResultCollector ResultCollector;
    const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
    SourceLocation SourceLocationBeg =
        clangd::getBeginningOfIdentifier(AST, Pos, SourceMgr.getMainFileID());
    tooling::RefactoringRuleContext Context(
        AST.getASTContext().getSourceManager());
    Context.setASTContext(AST.getASTContext());
    auto Rename = clang::tooling::RenameOccurrences::initiate(
        Context, SourceRange(SourceLocationBeg), NewName);
    if (!Rename)
      return CB(Rename.takeError());

    Rename->invoke(ResultCollector, Context);

    assert(ResultCollector.Result.hasValue());
    if (!ResultCollector.Result.getValue())
      return CB(ResultCollector.Result->takeError());

    std::vector<tooling::Replacement> Replacements;
    for (const tooling::AtomicChange &Change : ResultCollector.Result->get()) {
      tooling::Replacements ChangeReps = Change.getReplacements();
      for (const auto &Rep : ChangeReps) {
        // FIXME: Right now we only support renaming the main file, so we
        // drop replacements not for the main file. In the future, we might
        // consider to support:
        //   * rename in any included header
        //   * rename only in the "main" header
        //   * provide an error if there are symbols we won't rename (e.g.
        //     std::vector)
        //   * rename globally in project
        //   * rename in open files
        if (Rep.getFilePath() == File)
          Replacements.push_back(Rep);
      }
    }
    return CB(std::move(Replacements));
  };

  WorkScheduler.runWithAST(
      "Rename", File, Bind(Action, File.str(), NewName.str(), std::move(CB)));
}

void ClangdServer::dumpAST(PathRef File,
                           unique_function<void(std::string)> Callback) {
  auto Action = [](decltype(Callback) Callback, Expected<InputsAndAST> InpAST) {
    if (!InpAST) {
      llvm::consumeError(InpAST.takeError());
      return Callback("<no-ast>");
    }
    std::string Result;

    raw_string_ostream ResultOS(Result);
    clangd::dumpAST(InpAST->AST, ResultOS);
    ResultOS.flush();

    Callback(Result);
  };

  WorkScheduler.runWithAST("DumpAST", File, Bind(Action, std::move(Callback)));
}

void ClangdServer::findDefinitions(PathRef File, Position Pos,
                                   Callback<std::vector<Location>> CB) {
  auto Action = [Pos, this](Callback<std::vector<Location>> CB,
                            Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findDefinitions(InpAST->AST, Pos, Index));
  };

  WorkScheduler.runWithAST("Definitions", File, Bind(Action, std::move(CB)));
}

Optional<Path> ClangdServer::switchSourceHeader(PathRef Path) {

  StringRef SourceExtensions[] = {".cpp", ".c", ".cc", ".cxx",
                                  ".c++", ".m", ".mm"};
  StringRef HeaderExtensions[] = {".h", ".hh", ".hpp", ".hxx", ".inc"};

  StringRef PathExt = sys::path::extension(Path);

  // Lookup in a list of known extensions.
  auto SourceIter =
      llvm::find_if(SourceExtensions, [&PathExt](PathRef SourceExt) {
        return SourceExt.equals_lower(PathExt);
      });
  bool IsSource = SourceIter != std::end(SourceExtensions);

  auto HeaderIter =
      llvm::find_if(HeaderExtensions, [&PathExt](PathRef HeaderExt) {
        return HeaderExt.equals_lower(PathExt);
      });

  bool IsHeader = HeaderIter != std::end(HeaderExtensions);

  // We can only switch between the known extensions.
  if (!IsSource && !IsHeader)
    return None;

  // Array to lookup extensions for the switch. An opposite of where original
  // extension was found.
  ArrayRef<StringRef> NewExts;
  if (IsSource)
    NewExts = HeaderExtensions;
  else
    NewExts = SourceExtensions;

  // Storage for the new path.
  SmallString<128> NewPath = StringRef(Path);

  // Instance of vfs::FileSystem, used for file existence checks.
  auto FS = FSProvider.getFileSystem();

  // Loop through switched extension candidates.
  for (StringRef NewExt : NewExts) {
    sys::path::replace_extension(NewPath, NewExt);
    if (FS->exists(NewPath))
      return NewPath.str().str(); // First str() to convert from SmallString to
                                  // StringRef, second to convert from StringRef
                                  // to std::string

    // Also check NewExt in upper-case, just in case.
    sys::path::replace_extension(NewPath, NewExt.upper());
    if (FS->exists(NewPath))
      return NewPath.str().str();
  }

  return None;
}

Expected<tooling::Replacements>
ClangdServer::formatCode(StringRef Code, PathRef File,
                         ArrayRef<tooling::Range> Ranges) {
  // Call clang-format.
  auto FS = FSProvider.getFileSystem();
  auto Style = format::getStyle(format::DefaultFormatStyle, File,
                                format::DefaultFallbackStyle, Code, FS.get());
  if (!Style)
    return Style.takeError();

  tooling::Replacements IncludeReplaces =
      format::sortIncludes(*Style, Code, Ranges, File);
  auto Changed = tooling::applyAllReplacements(Code, IncludeReplaces);
  if (!Changed)
    return Changed.takeError();

  return IncludeReplaces.merge(format::reformat(
      Style.get(), *Changed,
      tooling::calculateRangesAfterReplacements(IncludeReplaces, Ranges),
      File));
}

void ClangdServer::findDocumentHighlights(
    PathRef File, Position Pos, Callback<std::vector<DocumentHighlight>> CB) {
  auto Action = [Pos](Callback<std::vector<DocumentHighlight>> CB,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findDocumentHighlights(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("Highlights", File, Bind(Action, std::move(CB)));
}

void ClangdServer::findHover(PathRef File, Position Pos,
                             Callback<Optional<Hover>> CB) {
  auto Action = [Pos](Callback<Optional<Hover>> CB,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getHover(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("Hover", File, Bind(Action, std::move(CB)));
}

tooling::CompileCommand ClangdServer::getCompileCommand(PathRef File) {
  trace::Span Span("GetCompileCommand");
  Optional<tooling::CompileCommand> C = CDB.getCompileCommand(File);
  if (!C) // FIXME: Suppress diagnostics? Let the user know?
    C = CDB.getFallbackCommand(File);

  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  C->CommandLine.push_back("-resource-dir=" + ResourceDir);
  return std::move(*C);
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

void ClangdServer::workspaceSymbols(
    StringRef Query, int Limit, Callback<std::vector<SymbolInformation>> CB) {
  std::string QueryCopy = Query;
  WorkScheduler.run(
      "getWorkspaceSymbols",
      Bind(
          [QueryCopy, Limit, this](decltype(CB) CB) {
            CB(clangd::getWorkspaceSymbols(QueryCopy, Limit, Index,
                                           WorkspaceRoot.getValueOr("")));
          },
          std::move(CB)));
}

void ClangdServer::documentSymbols(StringRef File,
                                   Callback<std::vector<DocumentSymbol>> CB) {
  auto Action = [](Callback<std::vector<DocumentSymbol>> CB,
                   llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getDocumentSymbols(InpAST->AST));
  };
  WorkScheduler.runWithAST("documentSymbols", File,
                           Bind(Action, std::move(CB)));
}

void ClangdServer::findReferences(PathRef File, Position Pos,
                                  Callback<std::vector<Location>> CB) {
  auto Action = [Pos, this](Callback<std::vector<Location>> CB,
                            Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findReferences(InpAST->AST, Pos, Index));
  };

  WorkScheduler.runWithAST("References", File, Bind(Action, std::move(CB)));
}

void ClangdServer::symbolInfo(PathRef File, Position Pos,
                              Callback<std::vector<SymbolDetails>> CB) {
  auto Action = [Pos](Callback<std::vector<SymbolDetails>> CB,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getSymbolInfo(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("SymbolInfo", File, Bind(Action, std::move(CB)));
}

std::vector<std::pair<Path, std::size_t>>
ClangdServer::getUsedBytesPerFile() const {
  return WorkScheduler.getUsedBytesPerFile();
}

LLVM_NODISCARD bool
ClangdServer::blockUntilIdleForTest(Optional<double> TimeoutSeconds) {
  return WorkScheduler.blockUntilIdle(timeoutSeconds(TimeoutSeconds)) &&
         (!BackgroundIdx ||
          BackgroundIdx->blockUntilIdleForTest(TimeoutSeconds));
}

} // namespace clangd
} // namespace clang
