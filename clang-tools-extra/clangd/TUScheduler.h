//===--- TUScheduler.h -------------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TUSCHEDULER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TUSCHEDULER_H

#include "ASTSignals.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "GlobalCompilationDatabase.h"
#include "clang-include-cleaner/Record.h"
#include "support/Function.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace clang {
namespace clangd {
class ParsedAST;
struct PreambleData;

/// Returns a number of a default async threads to use for TUScheduler.
/// Returned value is always >= 1 (i.e. will not cause requests to be processed
/// synchronously).
unsigned getDefaultAsyncThreadsCount();

struct InputsAndAST {
  const ParseInputs &Inputs;
  ParsedAST &AST;
};

struct InputsAndPreamble {
  llvm::StringRef Contents;
  const tooling::CompileCommand &Command;
  // This can be nullptr if no preamble is available.
  const PreambleData *Preamble;
  // This can be nullptr if no ASTSignals are available.
  const ASTSignals *Signals;
};

/// Determines whether diagnostics should be generated for a file snapshot.
enum class WantDiagnostics {
  Yes,  /// Diagnostics must be generated for this snapshot.
  No,   /// Diagnostics must not be generated for this snapshot.
  Auto, /// Diagnostics must be generated for this snapshot or a subsequent one,
        /// within a bounded amount of time.
};

/// Configuration of the AST retention policy. This only covers retention of
/// *idle* ASTs. If queue has operations requiring the AST, they might be
/// kept in memory.
struct ASTRetentionPolicy {
  /// Maximum number of ASTs to be retained in memory when there are no pending
  /// requests for them.
  unsigned MaxRetainedASTs = 3;
};

/// Clangd may wait after an update to see if another one comes along.
/// This is so we rebuild once the user stops typing, not when they start.
/// Debounce may be disabled/interrupted if we must build this version.
/// The debounce time is responsive to user preferences and rebuild time.
/// In the future, we could also consider different types of edits.
struct DebouncePolicy {
  using clock = std::chrono::steady_clock;

  /// The minimum time that we always debounce for.
  clock::duration Min = /*zero*/ {};
  /// The maximum time we may debounce for.
  clock::duration Max = /*zero*/ {};
  /// Target debounce, as a fraction of file rebuild time.
  /// e.g. RebuildRatio = 2, recent builds took 200ms => debounce for 400ms.
  float RebuildRatio = 1;

  /// Compute the time to debounce based on this policy and recent build times.
  clock::duration compute(llvm::ArrayRef<clock::duration> History) const;
  /// A policy that always returns the same duration, useful for tests.
  static DebouncePolicy fixed(clock::duration);
};

/// PreambleThrottler controls which preambles can build at any given time.
/// This can be used to limit overall concurrency, and to prioritize some
/// preambles over others.
/// In a distributed environment, a throttler may be able to coordinate resource
/// use across several clangd instances.
///
/// This class is threadsafe.
class PreambleThrottler {
public:
  virtual ~PreambleThrottler() = default;

  using RequestID = unsigned;
  using Callback = llvm::unique_function<void()>;
  /// Attempt to acquire resources to build a file's preamble.
  ///
  /// Does not block, may eventually invoke the callback to satisfy the request.
  /// If the callback is invoked, release() must be called afterwards.
  virtual RequestID acquire(llvm::StringRef Filename, Callback) = 0;
  /// Abandons the request/releases any resources that have been acquired.
  ///
  /// Must be called exactly once after acquire().
  /// acquire()'s callback will not be invoked after release() returns.
  virtual void release(RequestID) = 0;

  // FIXME: we may want to be able attach signals to filenames.
  //        this would allow the throttler to make better scheduling decisions.
};

enum class PreambleAction {
  Queued,
  Building,
  Idle,
};

struct ASTAction {
  enum Kind {
    Queued,        // The action is pending in the thread task queue to be run.
    RunningAction, // Started running actions on the TU.
    Building,      // The AST is being built.
    Idle, // Indicates the worker thread is idle, and ready to run any upcoming
          // actions.
  };
  ASTAction() = default;
  ASTAction(Kind K, llvm::StringRef Name) : K(K), Name(Name) {}
  Kind K = ASTAction::Idle;
  /// The name of the action currently running, e.g. Update, GoToDef, Hover.
  /// Empty if we are in the idle state.
  std::string Name;
};

// Internal status of the TU in TUScheduler.
struct TUStatus {
  struct BuildDetails {
    /// Indicates whether clang failed to build the TU.
    bool BuildFailed = false;
    /// Indicates whether we reused the prebuilt AST.
    bool ReuseAST = false;
  };
  /// Serialize this to an LSP file status item.
  FileStatus render(PathRef File) const;

  PreambleAction PreambleActivity = PreambleAction::Idle;
  ASTAction ASTActivity;
  /// Stores status of the last build for the translation unit.
  BuildDetails Details;
};

class ParsingCallbacks {
public:
  virtual ~ParsingCallbacks() = default;

  /// Called on the AST that was built for emitting the preamble. The built AST
  /// contains only AST nodes from the #include directives at the start of the
  /// file. AST node in the current file should be observed on onMainAST call.
  virtual void
  onPreambleAST(PathRef Path, llvm::StringRef Version, CapturedASTCtx Ctx,
                std::shared_ptr<const include_cleaner::PragmaIncludes>) {}
  /// The argument function is run under the critical section guarding against
  /// races when closing the files.
  using PublishFn = llvm::function_ref<void(llvm::function_ref<void()>)>;
  /// Called on the AST built for the file itself. Note that preamble AST nodes
  /// are not deserialized and should be processed in the onPreambleAST call
  /// instead.
  /// The \p AST always contains all AST nodes for the main file itself, and
  /// only a portion of the AST nodes deserialized from the preamble. Note that
  /// some nodes from the preamble may have been deserialized and may also be
  /// accessed from the main file AST, e.g. redecls of functions from preamble,
  /// etc. Clients are expected to process only the AST nodes from the main file
  /// in this callback (obtained via ParsedAST::getLocalTopLevelDecls) to obtain
  /// optimal performance.
  ///
  /// When information about the file (e.g. diagnostics) is
  /// published to clients, this should be wrapped in Publish, e.g.
  ///   void onMainAST(...) {
  ///     Diags = renderDiagnostics();
  ///     Publish([&] { notifyDiagnostics(Path, Diags); });
  ///   }
  /// This guarantees that clients will see results in the correct sequence if
  /// the file is concurrently closed and/or reopened. (The lambda passed to
  /// Publish() may never run in this case).
  virtual void onMainAST(PathRef Path, ParsedAST &AST, PublishFn Publish) {}

  /// Called whenever the AST fails to build. \p Diags will have the diagnostics
  /// that led to failure.
  virtual void onFailedAST(PathRef Path, llvm::StringRef Version,
                           std::vector<Diag> Diags, PublishFn Publish) {}

  /// Called whenever the TU status is updated.
  virtual void onFileUpdated(PathRef File, const TUStatus &Status) {}

  /// Preamble for the TU have changed. This might imply new semantics (e.g.
  /// different highlightings). Any actions on the file are guranteed to see new
  /// preamble after the callback.
  virtual void onPreamblePublished(PathRef File) {}
};

/// Handles running tasks for ClangdServer and managing the resources (e.g.,
/// preambles and ASTs) for opened files.
/// TUScheduler is not thread-safe, only one thread should be providing updates
/// and scheduling tasks.
/// Callbacks are run on a threadpool and it's appropriate to do slow work in
/// them. Each task has a name, used for tracing (should be UpperCamelCase).
class TUScheduler {
public:
  struct Options {
    /// Number of concurrent actions.
    /// Governs per-file worker threads and threads spawned for other tasks.
    /// (This does not prevent threads being spawned, but rather blocks them).
    /// If 0, executes actions synchronously on the calling thread.
    unsigned AsyncThreadsCount = getDefaultAsyncThreadsCount();

    /// Cache (large) preamble data in RAM rather than temporary files on disk.
    bool StorePreamblesInMemory = false;

    /// Time to wait after an update to see if another one comes along.
    /// This tries to ensure we rebuild once the user stops typing.
    DebouncePolicy UpdateDebounce;

    /// Determines when to keep idle ASTs in memory for future use.
    ASTRetentionPolicy RetentionPolicy;

    /// This throttler controls which preambles may be built at a given time.
    clangd::PreambleThrottler *PreambleThrottler = nullptr;

    /// Used to create a context that wraps each single operation.
    /// Typically to inject per-file configuration.
    /// If the path is empty, context sholud be "generic".
    std::function<Context(PathRef)> ContextProvider;
  };

  TUScheduler(const GlobalCompilationDatabase &CDB, const Options &Opts,
              std::unique_ptr<ParsingCallbacks> ASTCallbacks = nullptr);
  ~TUScheduler();

  struct FileStats {
    std::size_t UsedBytesAST = 0;
    std::size_t UsedBytesPreamble = 0;
    unsigned PreambleBuilds = 0;
    unsigned ASTBuilds = 0;
  };
  /// Returns resources used for each of the currently open files.
  /// Results are inherently racy as they measure activity of other threads.
  llvm::StringMap<FileStats> fileStats() const;

  /// Returns a list of files with ASTs currently stored in memory. This method
  /// is not very reliable and is only used for test. E.g., the results will not
  /// contain files that currently run something over their AST.
  std::vector<Path> getFilesWithCachedAST() const;

  /// Schedule an update for \p File.
  /// The compile command in \p Inputs is ignored; worker queries CDB to get
  /// the actual compile command.
  /// If diagnostics are requested (Yes), and the context is cancelled
  /// before they are prepared, they may be skipped if eventual-consistency
  /// permits it (i.e. WantDiagnostics is downgraded to Auto).
  /// Returns true if the file was not previously tracked.
  bool update(PathRef File, ParseInputs Inputs, WantDiagnostics WD);

  /// Remove \p File from the list of tracked files and schedule removal of its
  /// resources. Pending diagnostics for closed files may not be delivered, even
  /// if requested with WantDiags::Auto or WantDiags::Yes.
  void remove(PathRef File);

  /// Schedule an async task with no dependencies.
  /// Path may be empty (it is used only to set the Context).
  void run(llvm::StringRef Name, llvm::StringRef Path,
           llvm::unique_function<void()> Action);

  /// Similar to run, except the task is expected to be quick.
  /// This function will not honor AsyncThreadsCount (except
  /// if threading is disabled with AsyncThreadsCount=0)
  /// It is intended to run quick tasks that need to run ASAP
  void runQuick(llvm::StringRef Name, llvm::StringRef Path,
                llvm::unique_function<void()> Action);

  /// Defines how a runWithAST action is implicitly cancelled by other actions.
  enum ASTActionInvalidation {
    /// The request will run unless explicitly cancelled.
    NoInvalidation,
    /// The request will be implicitly cancelled by a subsequent update().
    /// (Only if the request was not yet cancelled).
    /// Useful for requests that are generated by clients, without any explicit
    /// user action. These can otherwise e.g. force every version to be built.
    InvalidateOnUpdate,
  };

  /// Schedule an async read of the AST. \p Action will be called when AST is
  /// ready. The AST passed to \p Action refers to the version of \p File
  /// tracked at the time of the call, even if new updates are received before
  /// \p Action is executed.
  /// If an error occurs during processing, it is forwarded to the \p Action
  /// callback.
  /// If the context is cancelled before the AST is ready, or the invalidation
  /// policy is triggered, the callback will receive a CancelledError.
  void runWithAST(llvm::StringRef Name, PathRef File,
                  Callback<InputsAndAST> Action,
                  ASTActionInvalidation = NoInvalidation);

  /// Controls whether preamble reads wait for the preamble to be up-to-date.
  enum PreambleConsistency {
    /// The preamble may be generated from an older version of the file.
    /// Reading from locations in the preamble may cause files to be re-read.
    /// This gives callers two options:
    /// - validate that the preamble is still valid, and only use it if so
    /// - accept that the preamble contents may be outdated, and try to avoid
    ///   reading source code from headers.
    /// This is the fastest option, usually a preamble is available immediately.
    Stale,
    /// Besides accepting stale preamble, this also allow preamble to be absent
    /// (not ready or failed to build).
    StaleOrAbsent,
  };

  /// Schedule an async read of the preamble.
  /// If there's no up-to-date preamble, we follow the PreambleConsistency
  /// policy.
  /// If an error occurs, it is forwarded to the \p Action callback.
  /// Context cancellation is ignored and should be handled by the Action.
  /// (In practice, the Action is almost always executed immediately).
  void runWithPreamble(llvm::StringRef Name, PathRef File,
                       PreambleConsistency Consistency,
                       Callback<InputsAndPreamble> Action);

  /// Wait until there are no scheduled or running tasks.
  /// Mostly useful for synchronizing tests.
  bool blockUntilIdle(Deadline D) const;

private:
  /// This class stores per-file data in the Files map.
  struct FileData;

public:
  /// Responsible for retaining and rebuilding idle ASTs. An implementation is
  /// an LRU cache.
  class ASTCache;
  /// Tracks headers included by open files, to get known-good compile commands.
  class HeaderIncluderCache;

  // The file being built/processed in the current thread. This is a hack in
  // order to get the file name into the index implementations. Do not depend on
  // this inside clangd.
  // FIXME: remove this when there is proper index support via build system
  // integration.
  // FIXME: move to ClangdServer via createProcessingContext.
  static std::optional<llvm::StringRef> getFileBeingProcessedInContext();

  void profile(MemoryTree &MT) const;

private:
  void runWithSemaphore(llvm::StringRef Name, llvm::StringRef Path,
                        llvm::unique_function<void()> Action, Semaphore &Sem);

  const GlobalCompilationDatabase &CDB;
  Options Opts;
  std::unique_ptr<ParsingCallbacks> Callbacks; // not nullptr
  Semaphore Barrier;
  Semaphore QuickRunBarrier;
  llvm::StringMap<std::unique_ptr<FileData>> Files;
  std::unique_ptr<ASTCache> IdleASTs;
  std::unique_ptr<HeaderIncluderCache> HeaderIncluders;
  // std::nullopt when running tasks synchronously and non-std::nullopt when
  // running tasks asynchronously.
  std::optional<AsyncTaskRunner> PreambleTasks;
  std::optional<AsyncTaskRunner> WorkerThreads;
  // Used to create contexts for operations that are not bound to a particular
  // file (e.g. index queries).
  std::string LastActiveFile;
};

} // namespace clangd
} // namespace clang

#endif
