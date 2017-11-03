//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
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

class FulfillPromiseGuard {
public:
  FulfillPromiseGuard(std::promise<void> &Promise) : Promise(Promise) {}

  ~FulfillPromiseGuard() { Promise.set_value(); }

private:
  std::promise<void> &Promise;
};

std::vector<tooling::Replacement> formatCode(StringRef Code, StringRef Filename,
                                             ArrayRef<tooling::Range> Ranges) {
  // Call clang-format.
  // FIXME: Don't ignore style.
  format::FormatStyle Style = format::getLLVMStyle();
  auto Result = format::reformat(Style, Code, Ranges, Filename);

  return std::vector<tooling::Replacement>(Result.begin(), Result.end());
}

std::string getStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

} // namespace

size_t clangd::positionToOffset(StringRef Code, Position P) {
  size_t Offset = 0;
  for (int I = 0; I != P.line; ++I) {
    // FIXME: \r\n
    // FIXME: UTF-8
    size_t F = Code.find('\n', Offset);
    if (F == StringRef::npos)
      return 0; // FIXME: Is this reasonable?
    Offset = F + 1;
  }
  return (Offset == 0 ? 0 : (Offset - 1)) + P.character;
}

/// Turn an offset in Code into a [line, column] pair.
Position clangd::offsetToPosition(StringRef Code, size_t Offset) {
  StringRef JustBefore = Code.substr(0, Offset);
  // FIXME: \r\n
  // FIXME: UTF-8
  int Lines = JustBefore.count('\n');
  int Cols = JustBefore.size() - JustBefore.rfind('\n') - 1;
  return {Lines, Cols};
}

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
                           clangd::CodeCompleteOptions CodeCompleteOpts,
                           clangd::Logger &Logger,
                           llvm::Optional<StringRef> ResourceDir)
    : Logger(Logger), CDB(CDB), DiagConsumer(DiagConsumer),
      FSProvider(FSProvider),
      ResourceDir(ResourceDir ? ResourceDir->str() : getStandardResourceDir()),
      PCHs(std::make_shared<PCHContainerOperations>()),
      CodeCompleteOpts(CodeCompleteOpts), WorkScheduler(AsyncThreadsCount) {}

void ClangdServer::setRootPath(PathRef RootPath) {
  std::string NewRootPath = llvm::sys::path::convert_to_slash(
      RootPath, llvm::sys::path::Style::posix);
  if (llvm::sys::fs::is_directory(NewRootPath))
    this->RootPath = NewRootPath;
}

std::future<void> ClangdServer::addDocument(PathRef File, StringRef Contents) {
  DocVersion Version = DraftMgr.updateDraft(File, Contents);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  std::shared_ptr<CppFile> Resources =
      Units.getOrCreateFile(File, ResourceDir, CDB, PCHs, Logger);
  return scheduleReparseAndDiags(File, VersionedDraft{Version, Contents.str()},
                                 std::move(Resources), std::move(TaggedFS));
}

std::future<void> ClangdServer::removeDocument(PathRef File) {
  DraftMgr.removeDraft(File);
  std::shared_ptr<CppFile> Resources = Units.removeIfPresent(File);
  return scheduleCancelRebuild(std::move(Resources));
}

std::future<void> ClangdServer::forceReparse(PathRef File) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft &&
         "forceReparse() was called for non-added document");

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  auto Recreated = Units.recreateFileIfCompileCommandChanged(File, ResourceDir,
                                                             CDB, PCHs, Logger);

  // Note that std::future from this cleanup action is ignored.
  scheduleCancelRebuild(std::move(Recreated.RemovedFile));
  // Schedule a reparse.
  return scheduleReparseAndDiags(File, std::move(FileContents),
                                 std::move(Recreated.FileInCollection),
                                 std::move(TaggedFS));
}

std::future<Tagged<std::vector<CompletionItem>>>
ClangdServer::codeComplete(PathRef File, Position Pos,
                           llvm::Optional<StringRef> OverridenContents,
                           IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  using ResultType = Tagged<std::vector<CompletionItem>>;

  std::promise<ResultType> ResultPromise;

  auto Callback = [](std::promise<ResultType> ResultPromise,
                     ResultType Result) -> void {
    ResultPromise.set_value(std::move(Result));
  };

  std::future<ResultType> ResultFuture = ResultPromise.get_future();
  codeComplete(BindWithForward(Callback, std::move(ResultPromise)), File, Pos,
               OverridenContents, UsedFS);
  return ResultFuture;
}

void ClangdServer::codeComplete(
    UniqueFunction<void(Tagged<std::vector<CompletionItem>>)> Callback,
    PathRef File, Position Pos, llvm::Optional<StringRef> OverridenContents,
    IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  using CallbackType =
      UniqueFunction<void(Tagged<std::vector<CompletionItem>>)>;

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
  // A task that will be run asynchronously.
  auto Task =
      // 'mutable' to reassign Preamble variable.
      [=](CallbackType Callback) mutable {
        if (!Preamble) {
          // Maybe we built some preamble before processing this request.
          Preamble = Resources->getPossiblyStalePreamble();
        }
        // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
        // both the old and the new version in case only one of them matches.

        std::vector<CompletionItem> Result = clangd::codeComplete(
            File, Resources->getCompileCommand(),
            Preamble ? &Preamble->Preamble : nullptr, Contents, Pos,
            TaggedFS.Value, PCHs, CodeCompleteOpts, Logger);

        Callback(make_tagged(std::move(Result), std::move(TaggedFS.Tag)));
      };

  WorkScheduler.addToFront(std::move(Task), std::move(Callback));
}

llvm::Expected<Tagged<SignatureHelp>>
ClangdServer::signatureHelp(PathRef File, Position Pos,
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
  auto Result = clangd::signatureHelp(File, Resources->getCompileCommand(),
                                      Preamble ? &Preamble->Preamble : nullptr,
                                      *OverridenContents, Pos, TaggedFS.Value,
                                      PCHs, Logger);
  return make_tagged(std::move(Result), TaggedFS.Tag);
}

std::vector<tooling::Replacement> ClangdServer::formatRange(PathRef File,
                                                            Range Rng) {
  std::string Code = getDocument(File);

  size_t Begin = positionToOffset(Code, Rng.start);
  size_t Len = positionToOffset(Code, Rng.end) - Begin;
  return formatCode(Code, File, {tooling::Range(Begin, Len)});
}

std::vector<tooling::Replacement> ClangdServer::formatFile(PathRef File) {
  // Format everything.
  std::string Code = getDocument(File);
  return formatCode(Code, File, {tooling::Range(0, Code.size())});
}

std::vector<tooling::Replacement> ClangdServer::formatOnType(PathRef File,
                                                             Position Pos) {
  // Look for the previous opening brace from the character position and
  // format starting from there.
  std::string Code = getDocument(File);
  size_t CursorPos = positionToOffset(Code, Pos);
  size_t PreviousLBracePos = StringRef(Code).find_last_of('{', CursorPos);
  if (PreviousLBracePos == StringRef::npos)
    PreviousLBracePos = CursorPos;
  size_t Len = 1 + CursorPos - PreviousLBracePos;

  return formatCode(Code, File, {tooling::Range(PreviousLBracePos, Len)});
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
ClangdServer::findDefinitions(PathRef File, Position Pos) {
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  if (!Resources)
    return llvm::make_error<llvm::StringError>(
        "findDefinitions called on non-added file",
        llvm::errc::invalid_argument);

  std::vector<Location> Result;
  Resources->getAST().get()->runUnderLock([Pos, &Result, this](ParsedAST *AST) {
    if (!AST)
      return;
    Result = clangd::findDefinitions(*AST, Pos, Logger);
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

std::future<void> ClangdServer::scheduleReparseAndDiags(
    PathRef File, VersionedDraft Contents, std::shared_ptr<CppFile> Resources,
    Tagged<IntrusiveRefCntPtr<vfs::FileSystem>> TaggedFS) {

  assert(Contents.Draft && "Draft must have contents");
  UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>()>
      DeferredRebuild =
          Resources->deferRebuild(*Contents.Draft, TaggedFS.Value);
  std::promise<void> DonePromise;
  std::future<void> DoneFuture = DonePromise.get_future();

  DocVersion Version = Contents.Version;
  Path FileStr = File;
  VFSTag Tag = TaggedFS.Tag;
  auto ReparseAndPublishDiags =
      [this, FileStr, Version,
       Tag](UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>()>
                DeferredRebuild,
            std::promise<void> DonePromise) -> void {
    FulfillPromiseGuard Guard(DonePromise);

    auto CurrentVersion = DraftMgr.getVersion(FileStr);
    if (CurrentVersion != Version)
      return; // This request is outdated

    auto Diags = DeferredRebuild();
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
                           std::move(DeferredRebuild), std::move(DonePromise));
  return DoneFuture;
}

std::future<void>
ClangdServer::scheduleCancelRebuild(std::shared_ptr<CppFile> Resources) {
  std::promise<void> DonePromise;
  std::future<void> DoneFuture = DonePromise.get_future();
  if (!Resources) {
    // No need to schedule any cleanup.
    DonePromise.set_value();
    return DoneFuture;
  }

  UniqueFunction<void()> DeferredCancel = Resources->deferCancelRebuild();
  auto CancelReparses = [Resources](std::promise<void> DonePromise,
                                    UniqueFunction<void()> DeferredCancel) {
    FulfillPromiseGuard Guard(DonePromise);
    DeferredCancel();
  };
  WorkScheduler.addToFront(std::move(CancelReparses), std::move(DonePromise),
                           std::move(DeferredCancel));
  return DoneFuture;
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}
