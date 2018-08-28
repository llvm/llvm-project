//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "Cancellation.h"
#include "CodeComplete.h"
#include "FindSymbols.h"
#include "Headers.h"
#include "SourceCode.h"
#include "Trace.h"
#include "XRefs.h"
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

using namespace clang;
using namespace clang::clangd;

namespace {

void ignoreError(llvm::Error Err) {
  handleAllErrors(std::move(Err), [](const llvm::ErrorInfoBase &) {});
}

std::string getStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

class RefactoringResultCollector final
    : public tooling::RefactoringResultConsumer {
public:
  void handleError(llvm::Error Err) override {
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
} // namespace

/// Manages dynamic index for open files. Each file might contribute two sets
/// of symbols to the dynamic index: symbols from the preamble and symbols
/// from the file itself. Those have different lifetimes and we merge results
/// from both
class ClangdServer::DynamicIndex : public ParsingCallbacks {
public:
  DynamicIndex(std::vector<std::string> URISchemes)
      : PreambleIdx(URISchemes), MainFileIdx(URISchemes),
        MergedIndex(mergeIndex(&MainFileIdx, &PreambleIdx)) {}

  SymbolIndex &index() const { return *MergedIndex; }

  void onPreambleAST(PathRef Path, ASTContext &Ctx,
                     std::shared_ptr<clang::Preprocessor> PP) override {
    PreambleIdx.update(Path, &Ctx, PP, /*TopLevelDecls=*/llvm::None);
  }

  void onMainAST(PathRef Path, ParsedAST &AST) override {

    MainFileIdx.update(Path, &AST.getASTContext(), AST.getPreprocessorPtr(),
                       AST.getLocalTopLevelDecls());
  }

private:
  FileIndex PreambleIdx;
  FileIndex MainFileIdx;
  /// Merged view into both indexes. Merges are performed in a similar manner
  /// to the merges of dynamic and static index.
  std::unique_ptr<SymbolIndex> MergedIndex;
};

ClangdServer::Options ClangdServer::optsForTest() {
  ClangdServer::Options Opts;
  Opts.UpdateDebounce = std::chrono::steady_clock::duration::zero(); // Faster!
  Opts.StorePreamblesInMemory = true;
  Opts.AsyncThreadsCount = 4; // Consistent!
  return Opts;
}

ClangdServer::ClangdServer(GlobalCompilationDatabase &CDB,
                           FileSystemProvider &FSProvider,
                           DiagnosticsConsumer &DiagConsumer,
                           const Options &Opts)
    : CDB(CDB), DiagConsumer(DiagConsumer), FSProvider(FSProvider),
      ResourceDir(Opts.ResourceDir ? Opts.ResourceDir->str()
                                   : getStandardResourceDir()),
      DynamicIdx(Opts.BuildDynamicSymbolIndex
                     ? new DynamicIndex(Opts.URISchemes)
                     : nullptr),
      PCHs(std::make_shared<PCHContainerOperations>()),
      // Pass a callback into `WorkScheduler` to extract symbols from a newly
      // parsed file and rebuild the file index synchronously each time an AST
      // is parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      WorkScheduler(Opts.AsyncThreadsCount, Opts.StorePreamblesInMemory,
                    DynamicIdx ? *DynamicIdx : noopParsingCallbacks(),
                    Opts.UpdateDebounce, Opts.RetentionPolicy) {
  if (DynamicIdx && Opts.StaticIndex) {
    MergedIndex = mergeIndex(&DynamicIdx->index(), Opts.StaticIndex);
    Index = MergedIndex.get();
  } else if (DynamicIdx)
    Index = &DynamicIdx->index();
  else if (Opts.StaticIndex)
    Index = Opts.StaticIndex;
  else
    Index = nullptr;
}

// Destructor has to be in .cpp file to see the definition of
// ClangdServer::DynamicIndex.
ClangdServer::~ClangdServer() = default;

void ClangdServer::setRootPath(PathRef RootPath) {
  auto FS = FSProvider.getFileSystem();
  auto Status = FS->status(RootPath);
  if (!Status)
    elog("Failed to get status for RootPath {0}: {1}", RootPath,
         Status.getError().message());
  else if (Status->isDirectory())
    this->RootPath = RootPath;
  else
    elog("The provided RootPath {0} is not a directory.", RootPath);
}

void ClangdServer::addDocument(PathRef File, StringRef Contents,
                               WantDiagnostics WantDiags) {
  DocVersion Version = ++InternalVersion[File];
  ParseInputs Inputs = {getCompileCommand(File), FSProvider.getFileSystem(),
                        Contents.str()};

  Path FileStr = File.str();
  WorkScheduler.update(File, std::move(Inputs), WantDiags,
                       [this, FileStr, Version](std::vector<Diag> Diags) {
                         consumeDiagnostics(FileStr, Version, std::move(Diags));
                       });
}

void ClangdServer::removeDocument(PathRef File) {
  ++InternalVersion[File];
  WorkScheduler.remove(File);
}

TaskHandle ClangdServer::codeComplete(PathRef File, Position Pos,
                                      const clangd::CodeCompleteOptions &Opts,
                                      Callback<CodeCompleteResult> CB) {
  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  TaskHandle TH = Task::createHandle();
  WithContext ContextWithCancellation(setCurrentTask(TH));
  // Copy PCHs to avoid accessing this->PCHs concurrently
  std::shared_ptr<PCHContainerOperations> PCHs = this->PCHs;
  auto FS = FSProvider.getFileSystem();

  auto Task = [PCHs, Pos, FS, CodeCompleteOpts,
               this](Path File, Callback<CodeCompleteResult> CB,
                     llvm::Expected<InputsAndPreamble> IP) {
    if (isCancelled())
      return CB(llvm::make_error<CancelledError>());

    if (!IP)
      return CB(IP.takeError());

    auto PreambleData = IP->Preamble;

    llvm::Optional<SpeculativeFuzzyFind> SpecFuzzyFind;
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
        File, IP->Command, PreambleData ? &PreambleData->Preamble : nullptr,
        PreambleData ? PreambleData->Includes : IncludeStructure(),
        IP->Contents, Pos, FS, PCHs, CodeCompleteOpts,
        SpecFuzzyFind ? SpecFuzzyFind.getPointer() : nullptr);
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

  WorkScheduler.runWithPreamble("CodeComplete", File,
                                Bind(Task, File.str(), std::move(CB)));
  return TH;
}

void ClangdServer::signatureHelp(PathRef File, Position Pos,
                                 Callback<SignatureHelp> CB) {

  auto PCHs = this->PCHs;
  auto FS = FSProvider.getFileSystem();
  auto *Index = this->Index;
  auto Action = [Pos, FS, PCHs, Index](Path File, Callback<SignatureHelp> CB,
                                       llvm::Expected<InputsAndPreamble> IP) {
    if (!IP)
      return CB(IP.takeError());

    auto PreambleData = IP->Preamble;
    CB(clangd::signatureHelp(File, IP->Command,
                             PreambleData ? &PreambleData->Preamble : nullptr,
                             IP->Contents, Pos, FS, PCHs, Index));
  };

  WorkScheduler.runWithPreamble("SignatureHelp", File,
                                Bind(Action, File.str(), std::move(CB)));
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatRange(StringRef Code, PathRef File, Range Rng) {
  llvm::Expected<size_t> Begin = positionToOffset(Code, Rng.start);
  if (!Begin)
    return Begin.takeError();
  llvm::Expected<size_t> End = positionToOffset(Code, Rng.end);
  if (!End)
    return End.takeError();
  return formatCode(Code, File, {tooling::Range(*Begin, *End - *Begin)});
}

llvm::Expected<tooling::Replacements> ClangdServer::formatFile(StringRef Code,
                                                               PathRef File) {
  // Format everything.
  return formatCode(Code, File, {tooling::Range(0, Code.size())});
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatOnType(StringRef Code, PathRef File, Position Pos) {
  // Look for the previous opening brace from the character position and
  // format starting from there.
  llvm::Expected<size_t> CursorPos = positionToOffset(Code, Pos);
  if (!CursorPos)
    return CursorPos.takeError();
  size_t PreviousLBracePos = StringRef(Code).find_last_of('{', *CursorPos);
  if (PreviousLBracePos == StringRef::npos)
    PreviousLBracePos = *CursorPos;
  size_t Len = *CursorPos - PreviousLBracePos;

  return formatCode(Code, File, {tooling::Range(PreviousLBracePos, Len)});
}

void ClangdServer::rename(PathRef File, Position Pos, llvm::StringRef NewName,
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
                           llvm::unique_function<void(std::string)> Callback) {
  auto Action = [](decltype(Callback) Callback,
                   llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST) {
      ignoreError(InpAST.takeError());
      return Callback("<no-ast>");
    }
    std::string Result;

    llvm::raw_string_ostream ResultOS(Result);
    clangd::dumpAST(InpAST->AST, ResultOS);
    ResultOS.flush();

    Callback(Result);
  };

  WorkScheduler.runWithAST("DumpAST", File, Bind(Action, std::move(Callback)));
}

void ClangdServer::findDefinitions(PathRef File, Position Pos,
                                   Callback<std::vector<Location>> CB) {
  auto Action = [Pos, this](Callback<std::vector<Location>> CB,
                            llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findDefinitions(InpAST->AST, Pos, Index));
  };

  WorkScheduler.runWithAST("Definitions", File, Bind(Action, std::move(CB)));
}

llvm::Optional<Path> ClangdServer::switchSourceHeader(PathRef Path) {

  StringRef SourceExtensions[] = {".cpp", ".c", ".cc", ".cxx",
                                  ".c++", ".m", ".mm"};
  StringRef HeaderExtensions[] = {".h", ".hh", ".hpp", ".hxx", ".inc"};

  StringRef PathExt = llvm::sys::path::extension(Path);

  // Lookup in a list of known extensions.
  auto SourceIter =
      std::find_if(std::begin(SourceExtensions), std::end(SourceExtensions),
                   [&PathExt](PathRef SourceExt) {
                     return SourceExt.equals_lower(PathExt);
                   });
  bool IsSource = SourceIter != std::end(SourceExtensions);

  auto HeaderIter =
      std::find_if(std::begin(HeaderExtensions), std::end(HeaderExtensions),
                   [&PathExt](PathRef HeaderExt) {
                     return HeaderExt.equals_lower(PathExt);
                   });

  bool IsHeader = HeaderIter != std::end(HeaderExtensions);

  // We can only switch between the known extensions.
  if (!IsSource && !IsHeader)
    return llvm::None;

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
    llvm::sys::path::replace_extension(NewPath, NewExt);
    if (FS->exists(NewPath))
      return NewPath.str().str(); // First str() to convert from SmallString to
                                  // StringRef, second to convert from StringRef
                                  // to std::string

    // Also check NewExt in upper-case, just in case.
    llvm::sys::path::replace_extension(NewPath, NewExt.upper());
    if (FS->exists(NewPath))
      return NewPath.str().str();
  }

  return llvm::None;
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatCode(llvm::StringRef Code, PathRef File,
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
                      llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findDocumentHighlights(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("Highlights", File, Bind(Action, std::move(CB)));
}

void ClangdServer::findHover(PathRef File, Position Pos,
                             Callback<llvm::Optional<Hover>> CB) {
  auto Action = [Pos](Callback<llvm::Optional<Hover>> CB,
                      llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getHover(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("Hover", File, Bind(Action, std::move(CB)));
}

void ClangdServer::consumeDiagnostics(PathRef File, DocVersion Version,
                                      std::vector<Diag> Diags) {
  // We need to serialize access to resulting diagnostics to avoid calling
  // `onDiagnosticsReady` in the wrong order.
  std::lock_guard<std::mutex> DiagsLock(DiagnosticsMutex);
  DocVersion &LastReportedDiagsVersion = ReportedDiagnosticVersions[File];

  // FIXME(ibiryukov): get rid of '<' comparison here. In the current
  // implementation diagnostics will not be reported after version counters'
  // overflow. This should not happen in practice, since DocVersion is a
  // 64-bit unsigned integer.
  if (Version < LastReportedDiagsVersion)
    return;
  LastReportedDiagsVersion = Version;

  DiagConsumer.onDiagnosticsReady(File, std::move(Diags));
}

tooling::CompileCommand ClangdServer::getCompileCommand(PathRef File) {
  trace::Span Span("GetCompileCommand");
  llvm::Optional<tooling::CompileCommand> C = CDB.getCompileCommand(File);
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
  CB(clangd::getWorkspaceSymbols(Query, Limit, Index,
                                 RootPath ? *RootPath : ""));
}

void ClangdServer::documentSymbols(
    StringRef File, Callback<std::vector<SymbolInformation>> CB) {
  auto Action = [](Callback<std::vector<SymbolInformation>> CB,
                   llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getDocumentSymbols(InpAST->AST));
  };
  WorkScheduler.runWithAST("documentSymbols", File,
                           Bind(Action, std::move(CB)));
}

std::vector<std::pair<Path, std::size_t>>
ClangdServer::getUsedBytesPerFile() const {
  return WorkScheduler.getUsedBytesPerFile();
}

LLVM_NODISCARD bool
ClangdServer::blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds) {
  return WorkScheduler.blockUntilIdle(timeoutSeconds(TimeoutSeconds));
}
