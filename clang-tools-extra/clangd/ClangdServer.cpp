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

WorkerRequest::WorkerRequest(WorkerRequestKind Kind, Path File,
                             DocVersion Version)
    : Kind(Kind), File(File), Version(Version) {}

ClangdScheduler::ClangdScheduler(ClangdServer &Server, bool RunSynchronously)
    : RunSynchronously(RunSynchronously) {
  if (RunSynchronously) {
    // Don't start the worker thread if we're running synchronously
    return;
  }

  // Initialize Worker in ctor body, rather than init list to avoid potentially
  // using not-yet-initialized members
  Worker = std::thread([&Server, this]() {
    while (true) {
      WorkerRequest Request;

      // Pick request from the queue
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        // Wait for more requests.
        RequestCV.wait(Lock, [this] { return !RequestQueue.empty() || Done; });
        if (Done)
          return;

        assert(!RequestQueue.empty() && "RequestQueue was empty");

        Request = std::move(RequestQueue.back());
        RequestQueue.pop_back();

        // Skip outdated requests
        if (Request.Version != Server.DraftMgr.getVersion(Request.File)) {
          // FIXME(ibiryukov): Logging
          // Output.log("Version for " + Twine(Request.File) +
          //            " in request is outdated, skipping request\n");
          continue;
        }
      } // unlock Mutex

      Server.handleRequest(std::move(Request));
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
    RequestCV.notify_one();
  } // unlock Mutex
  Worker.join();
}

void ClangdScheduler::enqueue(ClangdServer &Server, WorkerRequest Request) {
  if (RunSynchronously) {
    Server.handleRequest(Request);
    return;
  }

  std::lock_guard<std::mutex> Lock(Mutex);
  RequestQueue.push_back(Request);
  RequestCV.notify_one();
}

ClangdServer::ClangdServer(std::unique_ptr<GlobalCompilationDatabase> CDB,
                           std::unique_ptr<DiagnosticsConsumer> DiagConsumer,
                           bool RunSynchronously)
    : CDB(std::move(CDB)), DiagConsumer(std::move(DiagConsumer)),
      PCHs(std::make_shared<PCHContainerOperations>()),
      WorkScheduler(*this, RunSynchronously) {}

void ClangdServer::addDocument(PathRef File, StringRef Contents) {
  DocVersion NewVersion = DraftMgr.updateDraft(File, Contents);
  WorkScheduler.enqueue(
      *this, WorkerRequest(WorkerRequestKind::ParseAndPublishDiagnostics, File,
                           NewVersion));
}

void ClangdServer::removeDocument(PathRef File) {
  auto NewVersion = DraftMgr.removeDraft(File);
  WorkScheduler.enqueue(
      *this, WorkerRequest(WorkerRequestKind::RemoveDocData, File, NewVersion));
}

std::vector<CompletionItem> ClangdServer::codeComplete(PathRef File,
                                                       Position Pos) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft && "codeComplete is called for non-added document");

  std::vector<CompletionItem> Result;
  Units.runOnUnitWithoutReparse(
      File, *FileContents.Draft, *CDB, PCHs, [&](ClangdUnit &Unit) {
        Result = Unit.codeComplete(*FileContents.Draft, Pos);
      });
  return Result;
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

void ClangdServer::handleRequest(WorkerRequest Request) {
  switch (Request.Kind) {
  case WorkerRequestKind::ParseAndPublishDiagnostics: {
    auto FileContents = DraftMgr.getDraft(Request.File);
    if (FileContents.Version != Request.Version)
      return; // This request is outdated, do nothing

    assert(FileContents.Draft &&
           "No contents inside a file that was scheduled for reparse");
    Units.runOnUnit(Request.File, *FileContents.Draft, *CDB, PCHs,
                    [&](ClangdUnit const &Unit) {
                      DiagConsumer->onDiagnosticsReady(
                          Request.File, Unit.getLocalDiagnostics());
                    });
    break;
  }
  case WorkerRequestKind::RemoveDocData:
    if (Request.Version != DraftMgr.getVersion(Request.File))
      return; // This request is outdated, do nothing

    Units.removeUnitIfPresent(Request.File);
    break;
  }
}
