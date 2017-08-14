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
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
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
    Workers.push_back(std::thread([this]() {
      while (true) {
        std::future<void> Request;

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

        Request.get();
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
                           llvm::Optional<StringRef> ResourceDir)
    : CDB(CDB), DiagConsumer(DiagConsumer), FSProvider(FSProvider),
      ResourceDir(ResourceDir ? ResourceDir->str() : getStandardResourceDir()),
      PCHs(std::make_shared<PCHContainerOperations>()),
      WorkScheduler(AsyncThreadsCount) {}

std::future<void> ClangdServer::addDocument(PathRef File, StringRef Contents) {
  DocVersion Version = DraftMgr.updateDraft(File, Contents);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  std::shared_ptr<CppFile> Resources =
      Units.getOrCreateFile(File, ResourceDir, CDB, PCHs, TaggedFS.Value);
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
  auto Recreated = Units.recreateFileIfCompileCommandChanged(
      File, ResourceDir, CDB, PCHs, TaggedFS.Value);

  // Note that std::future from this cleanup action is ignored.
  scheduleCancelRebuild(std::move(Recreated.RemovedFile));
  // Schedule a reparse.
  return scheduleReparseAndDiags(File, std::move(FileContents),
                                 std::move(Recreated.FileInCollection),
                                 std::move(TaggedFS));
}

Tagged<std::vector<CompletionItem>>
ClangdServer::codeComplete(PathRef File, Position Pos,
                           llvm::Optional<StringRef> OverridenContents,
                           IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  std::string DraftStorage;
  if (!OverridenContents) {
    auto FileContents = DraftMgr.getDraft(File);
    assert(FileContents.Draft &&
           "codeComplete is called for non-added document");

    DraftStorage = std::move(*FileContents.Draft);
    OverridenContents = DraftStorage;
  }

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  assert(Resources && "Calling completion on non-added file");

  auto Preamble = Resources->getPossiblyStalePreamble();
  std::vector<CompletionItem> Result =
      clangd::codeComplete(File, Resources->getCompileCommand(),
                           Preamble ? &Preamble->Preamble : nullptr,
                           *OverridenContents, Pos, TaggedFS.Value, PCHs);
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

Tagged<std::vector<Location>> ClangdServer::findDefinitions(PathRef File,
                                                            Position Pos) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft &&
         "findDefinitions is called for non-added document");

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  std::shared_ptr<CppFile> Resources = Units.getFile(File);
  assert(Resources && "Calling findDefinitions on non-added file");

  std::vector<Location> Result;
  Resources->getAST().get()->runUnderLock([Pos, &Result](ParsedAST *AST) {
    if (!AST)
      return;
    Result = clangd::findDefinitions(*AST, Pos);
  });
  return make_tagged(std::move(Result), TaggedFS.Tag);
}

std::future<void> ClangdServer::scheduleReparseAndDiags(
    PathRef File, VersionedDraft Contents, std::shared_ptr<CppFile> Resources,
    Tagged<IntrusiveRefCntPtr<vfs::FileSystem>> TaggedFS) {

  assert(Contents.Draft && "Draft must have contents");
  std::future<llvm::Optional<std::vector<DiagWithFixIts>>> DeferredRebuild =
      Resources->deferRebuild(*Contents.Draft, TaggedFS.Value);
  std::promise<void> DonePromise;
  std::future<void> DoneFuture = DonePromise.get_future();

  DocVersion Version = Contents.Version;
  Path FileStr = File;
  VFSTag Tag = TaggedFS.Tag;
  auto ReparseAndPublishDiags =
      [this, FileStr, Version,
       Tag](std::future<llvm::Optional<std::vector<DiagWithFixIts>>>
                DeferredRebuild,
            std::promise<void> DonePromise) -> void {
    FulfillPromiseGuard Guard(DonePromise);

    auto CurrentVersion = DraftMgr.getVersion(FileStr);
    if (CurrentVersion != Version)
      return; // This request is outdated

    auto Diags = DeferredRebuild.get();
    if (!Diags)
      return; // A new reparse was requested before this one completed.
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

  std::future<void> DeferredCancel = Resources->deferCancelRebuild();
  auto CancelReparses = [Resources](std::promise<void> DonePromise,
                                    std::future<void> DeferredCancel) {
    FulfillPromiseGuard Guard(DonePromise);
    DeferredCancel.get();
  };
  WorkScheduler.addToFront(std::move(CancelReparses), std::move(DonePromise),
                           std::move(DeferredCancel));
  return DoneFuture;
}
