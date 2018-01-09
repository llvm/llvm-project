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
#include "SourceCode.h"
#include "XRefs.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Refactoring/RefactoringResultConsumer.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <future>

using namespace clang;
using namespace clang::clangd;

namespace {

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

Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
RealFileSystemProvider::getTaggedFileSystem(PathRef File) {
  return make_tagged(vfs::getRealFileSystem(), VFSTag());
}

unsigned clangd::getDefaultAsyncThreadsCount() {
  unsigned HardwareConcurrency = std::thread::hardware_concurrency();
  // C++ standard says that hardware_concurrency()
  // may return 0, fallback to 1 worker thread in
  // that case.
  if (HardwareConcurrency == 0)
    return 1;
  return HardwareConcurrency;
}

ClangdScheduler::ClangdScheduler(unsigned AsyncThreadsCount)
    : RunSynchronously(AsyncThreadsCount == 0) {
  if (RunSynchronously) {
    // Don't start the worker thread if we're running synchronously
    return;
  }

  Workers.reserve(AsyncThreadsCount);
  for (unsigned I = 0; I < AsyncThreadsCount; ++I) {
    Workers.push_back(std::thread([this, I]() {
      llvm::set_thread_name(llvm::formatv("scheduler/{0}", I));
      while (true) {
        UniqueFunction<void()> Request;

        // Pick request from the queue
        {
          std::unique_lock<std::mutex> Lock(Mutex);
          // Wait for more requests.
          RequestCV.wait(Lock,
                         [this] { return !RequestQueue.empty() || Done; });
          if (Done)
            return;

          assert(!RequestQueue.empty() && "RequestQueue was empty");

          // We process requests starting from the front of the queue. Users of
          // ClangdScheduler have a way to prioritise their requests by putting
          // them to the either side of the queue (using either addToEnd or
          // addToFront).
          Request = std::move(RequestQueue.front());
          RequestQueue.pop_front();
        } // unlock Mutex

        Request();
      }
    }));
  }
}

ClangdScheduler::~ClangdScheduler() {
  if (RunSynchronously)
    return; // no worker thread is running in that case

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Wake up the worker thread
    Done = true;
  } // unlock Mutex
  RequestCV.notify_all();

  for (auto &Worker : Workers)
    Worker.join();
}

ClangdServer::ClangdServer(GlobalCompilationDatabase &CDB,
                           DiagnosticsConsumer &DiagConsumer,
                           FileSystemProvider &FSProvider,
                           unsigned AsyncThreadsCount,
                           bool StorePreamblesInMemory,
                           bool BuildDynamicSymbolIndex,
                           llvm::Optional<StringRef> ResourceDir)
    : CDB(CDB), DiagConsumer(DiagConsumer), FSProvider(FSProvider),
      FileIdx(BuildDynamicSymbolIndex ? new FileIndex() : nullptr),
      // Pass a callback into `Units` to extract symbols from a newly parsed
      // file and rebuild the file index synchronously each time an AST is
      // parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      Units(FileIdx
                ? [this](const Context &Ctx, PathRef Path,
                         ParsedAST *AST) { FileIdx->update(Ctx, Path, AST); }
                : ASTParsedCallback()),
      ResourceDir(ResourceDir ? ResourceDir->str() : getStandardResourceDir()),
      PCHs(std::make_shared<PCHContainerOperations>()),
      StorePreamblesInMemory(StorePreamblesInMemory),
      WorkScheduler(AsyncThreadsCount) {}

void ClangdServer::setRootPath(PathRef RootPath) {
  std::string NewRootPath = llvm::sys::path::convert_to_slash(
      RootPath, llvm::sys::path::Style::posix);
  if (llvm::sys::fs::is_directory(NewRootPath))
    this->RootPath = NewRootPath;
}

std::future<Context> ClangdServer::addDocument(Context Ctx, PathRef File,
                                               StringRef Contents) {
  DocVersion Version = DraftMgr.updateDraft(File, Contents);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  std::shared_ptr<CppFile> Resources = Units.getOrCreateFile(
      File, ResourceDir, CDB, StorePreamblesInMemory, PCHs);
  return scheduleReparseAndDiags(std::move(Ctx), File,
                                 VersionedDraft{Version, Contents.str()},
                                 std::move(Resources), std::move(TaggedFS));
}

std::future<Context> ClangdServer::removeDocument(Context Ctx, PathRef File) {
  DraftMgr.removeDraft(File);
  std::shared_ptr<CppFile> Resources = Units.removeIfPresent(File);
  return scheduleCancelRebuild(std::move(Ctx), std::move(Resources));
}

std::future<Context> ClangdServer::forceReparse(Context Ctx, PathRef File) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft &&
         "forceReparse() was called for non-added document");

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  auto Recreated = Units.recreateFileIfCompileCommandChanged(
      File, ResourceDir, CDB, StorePreamblesInMemory, PCHs);

  // Note that std::future from this cleanup action is ignored.
  scheduleCancelRebuild(Ctx.clone(), std::move(Recreated.RemovedFile));
  // Schedule a reparse.
  return scheduleReparseAndDiags(std::move(Ctx), File, std::move(FileContents),
                                 std::move(Recreated.FileInCollection),
                                 std::move(TaggedFS));
}

std::future<std::pair<Context, Tagged<CompletionList>>>
ClangdServer::codeComplete(Context Ctx, PathRef File, Position Pos,
                           const clangd::CodeCompleteOptions &Opts,
                           llvm::Optional<StringRef> OverridenContents,
                           IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  using ResultType = std::pair<Context, Tagged<CompletionList>>;

  std::promise<ResultType> ResultPromise;

  auto Callback = [](std::promise<ResultType> ResultPromise, Context Ctx,
                     Tagged<CompletionList> Result) -> void {
    ResultPromise.set_value({std::move(Ctx), std::move(Result)});
  };

  std::future<ResultType> ResultFuture = ResultPromise.get_future();
  codeComplete(std::move(Ctx), File, Pos, Opts,
               BindWithForward(Callback, std::move(ResultPromise)),
               OverridenContents, UsedFS);
  return ResultFuture;
}

void ClangdServer::codeComplete(
    Context Ctx, PathRef File, Position Pos,
    const clangd::CodeCompleteOptions &Opts,
    UniqueFunction<void(Context, Tagged<CompletionList>)> Callback,
    llvm::Optional<StringRef> OverridenContents,
    IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  using CallbackType = UniqueFunction<void(Context, Tagged<CompletionList>)>;

  std::string Contents;
  if (OverridenContents) {
    Contents = *OverridenContents;
  } else {
    auto FileContents = DraftMgr.getDraft(File);
    assert(FileContents.Draft &&
           "codeComplete is called for non-added document");

    Contents = std::move(*FileContents.Draft);
  }

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  assert(Resources && "Calling completion on non-added file");

  // Remember the current Preamble and use it when async task starts executing.
  // At the point when async task starts executing, we may have a different
  // Preamble in Resources. However, we assume the Preamble that we obtain here
  // is reusable in completion more often.
  std::shared_ptr<const PreambleData> Preamble =
      Resources->getPossiblyStalePreamble();
  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (FileIdx)
    CodeCompleteOpts.Index = FileIdx.get();

  // Copy File, as it is a PathRef that will go out of scope before Task is
  // executed.
  Path FileStr = File;
  // Copy PCHs to avoid accessing this->PCHs concurrently
  std::shared_ptr<PCHContainerOperations> PCHs = this->PCHs;
  // A task that will be run asynchronously.
  auto Task =
      // 'mutable' to reassign Preamble variable.
      [FileStr, Preamble, Resources, Contents, Pos, CodeCompleteOpts, TaggedFS,
       PCHs](Context Ctx, CallbackType Callback) mutable {
        if (!Preamble) {
          // Maybe we built some preamble before processing this request.
          Preamble = Resources->getPossiblyStalePreamble();
        }
        // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
        // both the old and the new version in case only one of them matches.

        CompletionList Result = clangd::codeComplete(
            Ctx, FileStr, Resources->getCompileCommand(),
            Preamble ? &Preamble->Preamble : nullptr, Contents, Pos,
            TaggedFS.Value, PCHs, CodeCompleteOpts);

        Callback(std::move(Ctx),
                 make_tagged(std::move(Result), std::move(TaggedFS.Tag)));
      };

  WorkScheduler.addToFront(std::move(Task), std::move(Ctx),
                           std::move(Callback));
}

llvm::Expected<Tagged<SignatureHelp>>
ClangdServer::signatureHelp(const Context &Ctx, PathRef File, Position Pos,
                            llvm::Optional<StringRef> OverridenContents,
                            IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  std::string DraftStorage;
  if (!OverridenContents) {
    auto FileContents = DraftMgr.getDraft(File);
    if (!FileContents.Draft)
      return llvm::make_error<llvm::StringError>(
          "signatureHelp is called for non-added document",
          llvm::errc::invalid_argument);

    DraftStorage = std::move(*FileContents.Draft);
    OverridenContents = DraftStorage;
  }

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  if (!Resources)
    return llvm::make_error<llvm::StringError>(
        "signatureHelp is called for non-added document",
        llvm::errc::invalid_argument);

  auto Preamble = Resources->getPossiblyStalePreamble();
  auto Result =
      clangd::signatureHelp(Ctx, File, Resources->getCompileCommand(),
                            Preamble ? &Preamble->Preamble : nullptr,
                            *OverridenContents, Pos, TaggedFS.Value, PCHs);
  return make_tagged(std::move(Result), TaggedFS.Tag);
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatRange(StringRef Code, PathRef File, Range Rng) {
  size_t Begin = positionToOffset(Code, Rng.start);
  size_t Len = positionToOffset(Code, Rng.end) - Begin;
  return formatCode(Code, File, {tooling::Range(Begin, Len)});
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
  size_t CursorPos = positionToOffset(Code, Pos);
  size_t PreviousLBracePos = StringRef(Code).find_last_of('{', CursorPos);
  if (PreviousLBracePos == StringRef::npos)
    PreviousLBracePos = CursorPos;
  size_t Len = CursorPos - PreviousLBracePos;

  return formatCode(Code, File, {tooling::Range(PreviousLBracePos, Len)});
}

Expected<std::vector<tooling::Replacement>>
ClangdServer::rename(const Context &Ctx, PathRef File, Position Pos,
                     llvm::StringRef NewName) {
  std::string Code = getDocument(File);
  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  RefactoringResultCollector ResultCollector;
  Resources->getAST().get()->runUnderLock([&](ParsedAST *AST) {
    const SourceManager &SourceMgr = AST->getASTContext().getSourceManager();
    const FileEntry *FE =
        SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
    if (!FE)
      return;
    SourceLocation SourceLocationBeg =
        clangd::getBeginningOfIdentifier(*AST, Pos, FE);
    tooling::RefactoringRuleContext Context(
        AST->getASTContext().getSourceManager());
    Context.setASTContext(AST->getASTContext());
    auto Rename = clang::tooling::RenameOccurrences::initiate(
        Context, SourceRange(SourceLocationBeg), NewName.str());
    if (!Rename) {
      ResultCollector.Result = Rename.takeError();
      return;
    }
    Rename->invoke(ResultCollector, Context);
  });
  assert(ResultCollector.Result.hasValue());
  if (!ResultCollector.Result.getValue())
    return ResultCollector.Result->takeError();

  std::vector<tooling::Replacement> Replacements;
  for (const tooling::AtomicChange &Change : ResultCollector.Result->get()) {
    tooling::Replacements ChangeReps = Change.getReplacements();
    for (const auto &Rep : ChangeReps) {
      // FIXME: Right now we only support renaming the main file, so we drop
      // replacements not for the main file. In the future, we might consider to
      // support:
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
  return Replacements;
}

std::string ClangdServer::getDocument(PathRef File) {
  auto draft = DraftMgr.getDraft(File);
  assert(draft.Draft && "File is not tracked, cannot get contents");
  return *draft.Draft;
}

std::string ClangdServer::dumpAST(PathRef File) {
  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  assert(Resources && "dumpAST is called for non-added document");

  std::string Result;
  Resources->getAST().get()->runUnderLock([&Result](ParsedAST *AST) {
    llvm::raw_string_ostream ResultOS(Result);
    if (AST) {
      clangd::dumpAST(*AST, ResultOS);
    } else {
      ResultOS << "<no-ast>";
    }
    ResultOS.flush();
  });
  return Result;
}

llvm::Expected<Tagged<std::vector<Location>>>
ClangdServer::findDefinitions(const Context &Ctx, PathRef File, Position Pos) {
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  if (!Resources)
    return llvm::make_error<llvm::StringError>(
        "findDefinitions called on non-added file",
        llvm::errc::invalid_argument);

  std::vector<Location> Result;
  Resources->getAST().get()->runUnderLock([Pos, &Result, &Ctx](ParsedAST *AST) {
    if (!AST)
      return;
    Result = clangd::findDefinitions(Ctx, *AST, Pos);
  });
  return make_tagged(std::move(Result), TaggedFS.Tag);
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

  // We can only switch between extensions known extensions.
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
  auto FS = FSProvider.getTaggedFileSystem(Path).Value;

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
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  auto StyleOrError =
      format::getStyle("file", File, "LLVM", Code, TaggedFS.Value.get());
  if (!StyleOrError) {
    return StyleOrError.takeError();
  } else {
    return format::reformat(StyleOrError.get(), Code, Ranges, File);
  }
}

llvm::Expected<Tagged<std::vector<DocumentHighlight>>>
ClangdServer::findDocumentHighlights(const Context &Ctx, PathRef File,
                                     Position Pos) {
  auto FileContents = DraftMgr.getDraft(File);
  if (!FileContents.Draft)
    return llvm::make_error<llvm::StringError>(
        "findDocumentHighlights called on non-added file",
        llvm::errc::invalid_argument);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  if (!Resources)
    return llvm::make_error<llvm::StringError>(
        "findDocumentHighlights called on non-added file",
        llvm::errc::invalid_argument);

  std::vector<DocumentHighlight> Result;
  llvm::Optional<llvm::Error> Err;
  Resources->getAST().get()->runUnderLock([Pos, &Ctx, &Err,
                                           &Result](ParsedAST *AST) {
    if (!AST) {
      Err = llvm::make_error<llvm::StringError>("Invalid AST",
                                                llvm::errc::invalid_argument);
      return;
    }
    Result = clangd::findDocumentHighlights(Ctx, *AST, Pos);
  });

  if (Err)
    return std::move(*Err);
  return make_tagged(Result, TaggedFS.Tag);
}

std::future<Context> ClangdServer::scheduleReparseAndDiags(
    Context Ctx, PathRef File, VersionedDraft Contents,
    std::shared_ptr<CppFile> Resources,
    Tagged<IntrusiveRefCntPtr<vfs::FileSystem>> TaggedFS) {

  assert(Contents.Draft && "Draft must have contents");
  UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>(const Context &)>
      DeferredRebuild =
          Resources->deferRebuild(*Contents.Draft, TaggedFS.Value);
  std::promise<Context> DonePromise;
  std::future<Context> DoneFuture = DonePromise.get_future();

  DocVersion Version = Contents.Version;
  Path FileStr = File;
  VFSTag Tag = TaggedFS.Tag;
  auto ReparseAndPublishDiags =
      [this, FileStr, Version,
       Tag](UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>(
                const Context &)>
                DeferredRebuild,
            std::promise<Context> DonePromise, Context Ctx) -> void {
    auto Guard = onScopeExit([&]() { DonePromise.set_value(std::move(Ctx)); });

    auto CurrentVersion = DraftMgr.getVersion(FileStr);
    if (CurrentVersion != Version)
      return; // This request is outdated

    auto Diags = DeferredRebuild(Ctx);
    if (!Diags)
      return; // A new reparse was requested before this one completed.

    // We need to serialize access to resulting diagnostics to avoid calling
    // `onDiagnosticsReady` in the wrong order.
    std::lock_guard<std::mutex> DiagsLock(DiagnosticsMutex);
    DocVersion &LastReportedDiagsVersion = ReportedDiagnosticVersions[FileStr];
    // FIXME(ibiryukov): get rid of '<' comparison here. In the current
    // implementation diagnostics will not be reported after version counters'
    // overflow. This should not happen in practice, since DocVersion is a
    // 64-bit unsigned integer.
    if (Version < LastReportedDiagsVersion)
      return;
    LastReportedDiagsVersion = Version;

    DiagConsumer.onDiagnosticsReady(FileStr,
                                    make_tagged(std::move(*Diags), Tag));
  };

  WorkScheduler.addToFront(std::move(ReparseAndPublishDiags),
                           std::move(DeferredRebuild), std::move(DonePromise),
                           std::move(Ctx));
  return DoneFuture;
}

std::future<Context>
ClangdServer::scheduleCancelRebuild(Context Ctx,
                                    std::shared_ptr<CppFile> Resources) {
  std::promise<Context> DonePromise;
  std::future<Context> DoneFuture = DonePromise.get_future();
  if (!Resources) {
    // No need to schedule any cleanup.
    DonePromise.set_value(std::move(Ctx));
    return DoneFuture;
  }

  UniqueFunction<void()> DeferredCancel = Resources->deferCancelRebuild();
  auto CancelReparses = [Resources](std::promise<Context> DonePromise,
                                    UniqueFunction<void()> DeferredCancel,
                                    Context Ctx) {
    DeferredCancel();
    DonePromise.set_value(std::move(Ctx));
  };
  WorkScheduler.addToFront(std::move(CancelReparses), std::move(DonePromise),
                           std::move(DeferredCancel), std::move(Ctx));
  return DoneFuture;
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}
