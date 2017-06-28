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

ClangdScheduler::ClangdScheduler(bool RunSynchronously)
    : RunSynchronously(RunSynchronously) {
  if (RunSynchronously) {
    // Don't start the worker thread if we're running synchronously
    return;
  }

  // Initialize Worker in ctor body, rather than init list to avoid potentially
  // using not-yet-initialized members
  Worker = std::thread([this]() {
    while (true) {
      std::function<void()> Request;

      // Pick request from the queue
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        // Wait for more requests.
        RequestCV.wait(Lock, [this] { return !RequestQueue.empty() || Done; });
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
  });
}

ClangdScheduler::~ClangdScheduler() {
  if (RunSynchronously)
    return; // no worker thread is running in that case

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Wake up the worker thread
    Done = true;
  } // unlock Mutex
  RequestCV.notify_one();
  Worker.join();
}

void ClangdScheduler::addToFront(std::function<void()> Request) {
  if (RunSynchronously) {
    Request();
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    RequestQueue.push_front(Request);
  }
  RequestCV.notify_one();
}

void ClangdScheduler::addToEnd(std::function<void()> Request) {
  if (RunSynchronously) {
    Request();
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    RequestQueue.push_back(Request);
  }
  RequestCV.notify_one();
}

ClangdServer::ClangdServer(GlobalCompilationDatabase &CDB,
                           DiagnosticsConsumer &DiagConsumer,
                           FileSystemProvider &FSProvider,
                           bool RunSynchronously,
                           llvm::Optional<StringRef> ResourceDir)
    : CDB(CDB), DiagConsumer(DiagConsumer), FSProvider(FSProvider),
      ResourceDir(ResourceDir ? ResourceDir->str() : getStandardResourceDir()),
      PCHs(std::make_shared<PCHContainerOperations>()),
      WorkScheduler(RunSynchronously) {}

void ClangdServer::addDocument(PathRef File, StringRef Contents) {
  DocVersion Version = DraftMgr.updateDraft(File, Contents);
  Path FileStr = File;
  WorkScheduler.addToFront([this, FileStr, Version]() {
    auto FileContents = DraftMgr.getDraft(FileStr);
    if (FileContents.Version != Version)
      return; // This request is outdated, do nothing

    assert(FileContents.Draft &&
           "No contents inside a file that was scheduled for reparse");
    auto TaggedFS = FSProvider.getTaggedFileSystem(FileStr);
    Units.runOnUnit(
        FileStr, *FileContents.Draft, ResourceDir, CDB, PCHs, TaggedFS.Value,
        [&](ClangdUnit const &Unit) {
          DiagConsumer.onDiagnosticsReady(
              FileStr, make_tagged(Unit.getLocalDiagnostics(), TaggedFS.Tag));
        });
  });
}

void ClangdServer::removeDocument(PathRef File) {
  auto Version = DraftMgr.removeDraft(File);
  Path FileStr = File;
  WorkScheduler.addToFront([this, FileStr, Version]() {
    if (Version != DraftMgr.getVersion(FileStr))
      return; // This request is outdated, do nothing

    Units.removeUnitIfPresent(FileStr);
  });
}

void ClangdServer::forceReparse(PathRef File) {
  // The addDocument schedules the reparse even if the contents of the file
  // never changed, so we just call it here.
  addDocument(File, getDocument(File));
}

Tagged<std::vector<CompletionItem>>
ClangdServer::codeComplete(PathRef File, Position Pos,
                           llvm::Optional<StringRef> OverridenContents) {
  std::string DraftStorage;
  if (!OverridenContents) {
    auto FileContents = DraftMgr.getDraft(File);
    assert(FileContents.Draft &&
           "codeComplete is called for non-added document");

    DraftStorage = std::move(*FileContents.Draft);
    OverridenContents = DraftStorage;
  }

  std::vector<CompletionItem> Result;
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  // It would be nice to use runOnUnitWithoutReparse here, but we can't
  // guarantee the correctness of code completion cache here if we don't do the
  // reparse.
  Units.runOnUnit(File, *OverridenContents, ResourceDir, CDB, PCHs,
                  TaggedFS.Value, [&](ClangdUnit &Unit) {
                    Result = Unit.codeComplete(*OverridenContents, Pos,
                                               TaggedFS.Value);
                  });
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
  std::promise<std::string> DumpPromise;
  auto DumpFuture = DumpPromise.get_future();
  auto Version = DraftMgr.getVersion(File);

  WorkScheduler.addToEnd([this, &DumpPromise, File, Version]() {
    assert(DraftMgr.getVersion(File) == Version && "Version has changed");

    Units.runOnExistingUnit(File, [&DumpPromise](ClangdUnit &Unit) {
      std::string Result;

      llvm::raw_string_ostream ResultOS(Result);
      Unit.dumpAST(ResultOS);
      ResultOS.flush();

      DumpPromise.set_value(std::move(Result));
    });
  });
  return DumpFuture.get();
}

Tagged<std::vector<Location>>
ClangdServer::findDefinitions(PathRef File, Position Pos) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft && "findDefinitions is called for non-added document");

  std::vector<Location> Result;
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  Units.runOnUnit(File, *FileContents.Draft, ResourceDir, CDB, PCHs,
      TaggedFS.Value, [&](ClangdUnit &Unit) {
        Result = Unit.findDefinitions(Pos);
      });
  return make_tagged(std::move(Result), TaggedFS.Tag);
}
