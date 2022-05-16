//===--- Background.h - Build an index in a background thread ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUND_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUND_H

#include "GlobalCompilationDatabase.h"
#include "SourceCode.h"
#include "index/BackgroundRebuild.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/Serialization.h"
#include "support/Context.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "support/ThreadsafeFS.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Threading.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace clang {
namespace clangd {

// Handles storage and retrieval of index shards. Both store and load
// operations can be called from multiple-threads concurrently.
class BackgroundIndexStorage {
public:
  virtual ~BackgroundIndexStorage() = default;

  // Shards of the index are stored and retrieved independently, keyed by shard
  // identifier - in practice this is a source file name
  virtual llvm::Error storeShard(llvm::StringRef ShardIdentifier,
                                 IndexFileOut Shard) const = 0;

  // Tries to load shard with given identifier, returns nullptr if shard
  // couldn't be loaded.
  virtual std::unique_ptr<IndexFileIn>
  loadShard(llvm::StringRef ShardIdentifier) const = 0;

  // The factory provides storage for each File.
  // It keeps ownership of the storage instances, and should manage caching
  // itself. Factory must be threadsafe and never returns nullptr.
  using Factory = llvm::unique_function<BackgroundIndexStorage *(PathRef)>;

  // Creates an Index Storage that saves shards into disk. Index storage uses
  // CDBDirectory + ".cache/clangd/index/" as the folder to save shards.
  // CDBDirectory is the first directory containing a CDB in parent directories
  // of a file, or user cache directory if none was found, e.g. stdlib headers.
  static Factory createDiskBackedStorageFactory(
      std::function<llvm::Optional<ProjectInfo>(PathRef)> GetProjectInfo);
};

// A priority queue of tasks which can be run on (external) worker threads.
class BackgroundQueue {
public:
  /// A work item on the thread pool's queue.
  struct Task {
    explicit Task(std::function<void()> Run) : Run(std::move(Run)) {}

    std::function<void()> Run;
    llvm::ThreadPriority ThreadPri = llvm::ThreadPriority::Low;
    unsigned QueuePri = 0; // Higher-priority tasks will run first.
    std::string Tag;       // Allows priority to be boosted later.
    uint64_t Key = 0;      // If the key matches a previous task, drop this one.
                           // (in practice this means we never reindex a file).

    bool operator<(const Task &O) const { return QueuePri < O.QueuePri; }
  };

  // Describes the number of tasks processed by the queue.
  struct Stats {
    unsigned Enqueued = 0;  // Total number of tasks ever enqueued.
    unsigned Active = 0;    // Tasks being currently processed by a worker.
    unsigned Completed = 0; // Tasks that have been finished.
    unsigned LastIdle = 0;  // Number of completed tasks when last empty.
  };

  BackgroundQueue(std::function<void(Stats)> OnProgress = nullptr)
      : OnProgress(OnProgress) {}

  // Add tasks to the queue.
  void push(Task);
  void append(std::vector<Task>);
  // Boost priority of current and new tasks with matching Tag, if they are
  // lower priority.
  // Reducing the boost of a tag affects future tasks but not current ones.
  void boost(llvm::StringRef Tag, unsigned NewPriority);

  // Process items on the queue until the queue is stopped.
  // If the queue becomes empty, OnIdle will be called (on one worker).
  void work(std::function<void()> OnIdle = nullptr);

  // Stop processing new tasks, allowing all work() calls to return soon.
  void stop();

  // Disables thread priority lowering to ensure progress on loaded systems.
  // Only affects tasks that run after the call.
  static void preventThreadStarvationInTests();
  LLVM_NODISCARD bool
  blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds);

private:
  void notifyProgress() const; // Requires lock Mu
  bool adjust(Task &T);

  std::mutex Mu;
  Stats Stat;
  std::condition_variable CV;
  bool ShouldStop = false;
  std::vector<Task> Queue; // max-heap
  llvm::StringMap<unsigned> Boosts;
  std::function<void(Stats)> OnProgress;
  llvm::DenseSet<uint64_t> SeenKeys;
};

// Builds an in-memory index by by running the static indexer action over
// all commands in a compilation database. Indexing happens in the background.
// FIXME: it should watch for changes to files on disk.
class BackgroundIndex : public SwapIndex {
public:
  struct Options {
    // Arbitrary value to ensure some concurrency in tests.
    // In production an explicit value is specified.
    size_t ThreadPoolSize = 4;
    // Callback that provides notifications as indexing makes progress.
    std::function<void(BackgroundQueue::Stats)> OnProgress = nullptr;
    // Function called to obtain the Context to use while indexing the specified
    // file. Called with the empty string for other tasks.
    // (When called, the context from BackgroundIndex construction is active).
    std::function<Context(PathRef)> ContextProvider = nullptr;
  };

  /// Creates a new background index and starts its threads.
  /// The current Context will be propagated to each worker thread.
  BackgroundIndex(const ThreadsafeFS &, const GlobalCompilationDatabase &CDB,
                  BackgroundIndexStorage::Factory IndexStorageFactory,
                  Options Opts);
  ~BackgroundIndex(); // Blocks while the current task finishes.

  // Enqueue translation units for indexing.
  // The indexing happens in a background thread, so the symbols will be
  // available sometime later.
  void enqueue(const std::vector<std::string> &ChangedFiles) {
    Queue.push(changedFilesTask(ChangedFiles));
  }

  /// Boosts priority of indexing related to Path.
  /// Typically used to index TUs when headers are opened.
  void boostRelated(llvm::StringRef Path);

  // Cause background threads to stop after ther current task, any remaining
  // tasks will be discarded.
  void stop() {
    Rebuilder.shutdown();
    Queue.stop();
  }

  // Wait until the queue is empty, to allow deterministic testing.
  LLVM_NODISCARD bool
  blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds = 10) {
    return Queue.blockUntilIdleForTest(TimeoutSeconds);
  }

  void profile(MemoryTree &MT) const;

private:
  /// Represents the state of a single file when indexing was performed.
  struct ShardVersion {
    FileDigest Digest{{0}};
    bool HadErrors = false;
  };

  /// Given index results from a TU, only update symbols coming from files with
  /// different digests than \p ShardVersionsSnapshot. Also stores new index
  /// information on IndexStorage.
  void update(llvm::StringRef MainFile, IndexFileIn Index,
              const llvm::StringMap<ShardVersion> &ShardVersionsSnapshot,
              bool HadErrors);

  // configuration
  const ThreadsafeFS &TFS;
  const GlobalCompilationDatabase &CDB;
  std::function<Context(PathRef)> ContextProvider;

  llvm::Error index(tooling::CompileCommand);

  FileSymbols IndexedSymbols;
  BackgroundIndexRebuilder Rebuilder;
  llvm::StringMap<ShardVersion> ShardVersions; // Key is absolute file path.
  std::mutex ShardVersionsMu;

  BackgroundIndexStorage::Factory IndexStorageFactory;
  // Tries to load shards for the MainFiles and their dependencies.
  std::vector<std::string> loadProject(std::vector<std::string> MainFiles);

  BackgroundQueue::Task
  changedFilesTask(const std::vector<std::string> &ChangedFiles);
  BackgroundQueue::Task indexFileTask(std::string Path);

  // from lowest to highest priority
  enum QueuePriority {
    IndexFile,
    IndexBoostedFile,
    LoadShards,
  };
  BackgroundQueue Queue;
  AsyncTaskRunner ThreadPool;
  GlobalCompilationDatabase::CommandChanged::Subscription CommandsChanged;
};

} // namespace clangd
} // namespace clang

#endif
