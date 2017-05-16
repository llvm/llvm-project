//===--- ClangdServer.h - Main clangd server code ----------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDSERVER_H

#include "ClangdUnitStore.h"
#include "DraftStore.h"
#include "GlobalCompilationDatabase.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

#include "ClangdUnit.h"
#include "Protocol.h"

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace clang {
class PCHContainerOperations;

namespace clangd {

/// Turn a [line, column] pair into an offset in Code.
size_t positionToOffset(StringRef Code, Position P);

/// Turn an offset in Code into a [line, column] pair.
Position offsetToPosition(StringRef Code, size_t Offset);

class DiagnosticsConsumer {
public:
  virtual ~DiagnosticsConsumer() = default;

  /// Called by ClangdServer when \p Diagnostics for \p File are ready.
  virtual void onDiagnosticsReady(PathRef File,
                                  std::vector<DiagWithFixIts> Diagnostics) = 0;
};

enum class WorkerRequestKind { ParseAndPublishDiagnostics, RemoveDocData };

/// A request to the worker thread
class WorkerRequest {
public:
  WorkerRequest() = default;
  WorkerRequest(WorkerRequestKind Kind, Path File, DocVersion Version);

  WorkerRequestKind Kind;
  Path File;
  DocVersion Version;
};

class ClangdServer;

/// Handles running WorkerRequests of ClangdServer on a separate threads.
/// Currently runs only one worker thread.
class ClangdScheduler {
public:
  ClangdScheduler(ClangdServer &Server, bool RunSynchronously);
  ~ClangdScheduler();

  /// Enqueue WorkerRequest to be run on a worker thread
  void enqueue(ClangdServer &Server, WorkerRequest Request);

private:
  bool RunSynchronously;
  std::mutex Mutex;
  /// We run some tasks on a separate thread(parsing, ClangdUnit cleanup).
  /// This thread looks into RequestQueue to find requests to handle and
  /// terminates when Done is set to true.
  std::thread Worker;
  /// Setting Done to true will make the worker thread terminate.
  bool Done = false;
  /// A LIFO queue of requests. Note that requests are discarded if the
  /// `version` field is not equal to the one stored inside DraftStore.
  /// FIXME(krasimir): code completion should always have priority over parsing
  /// for diagnostics.
  std::deque<WorkerRequest> RequestQueue;
  /// Condition variable to wake up the worker thread.
  std::condition_variable RequestCV;
};

/// Provides API to manage ASTs for a collection of C++ files and request
/// various language features(currently, only codeCompletion and async
/// diagnostics for tracked files).
class ClangdServer {
public:
  ClangdServer(std::unique_ptr<GlobalCompilationDatabase> CDB,
               std::unique_ptr<DiagnosticsConsumer> DiagConsumer,
               bool RunSynchronously);

  /// Add a \p File to the list of tracked C++ files or update the contents if
  /// \p File is already tracked. Also schedules parsing of the AST for it on a
  /// separate thread. When the parsing is complete, DiagConsumer passed in
  /// constructor will receive onDiagnosticsReady callback.
  void addDocument(PathRef File, StringRef Contents);
  /// Remove \p File from list of tracked files, schedule a request to free
  /// resources associated with it.
  void removeDocument(PathRef File);

  /// Run code completion for \p File at \p Pos.
  std::vector<CompletionItem> codeComplete(PathRef File, Position Pos);

  /// Run formatting for \p Rng inside \p File.
  std::vector<tooling::Replacement> formatRange(PathRef File, Range Rng);
  /// Run formatting for the whole \p File.
  std::vector<tooling::Replacement> formatFile(PathRef File);
  /// Run formatting after a character was typed at \p Pos in \p File.
  std::vector<tooling::Replacement> formatOnType(PathRef File, Position Pos);

  /// Gets current document contents for \p File. \p File must point to a
  /// currently tracked file.
  /// FIXME(ibiryukov): This function is here to allow offset-to-Position
  /// conversions in outside code, maybe there's a way to get rid of it.
  std::string getDocument(PathRef File);

private:
  friend class ClangdScheduler;

  /// This function is called on a worker thread.
  void handleRequest(WorkerRequest Request);

  std::unique_ptr<GlobalCompilationDatabase> CDB;
  std::unique_ptr<DiagnosticsConsumer> DiagConsumer;
  DraftStore DraftMgr;
  ClangdUnitStore Units;
  std::shared_ptr<PCHContainerOperations> PCHs;
  // WorkScheduler has to be the last member, because its destructor has to be
  // called before all other members to stop the worker thread that references
  // ClangdServer
  ClangdScheduler WorkScheduler;
};

} // namespace clangd
} // namespace clang

#endif
