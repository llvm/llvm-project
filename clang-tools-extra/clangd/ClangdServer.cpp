//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace clang::clangd;

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
