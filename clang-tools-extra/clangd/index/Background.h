//===--- Background.h - Build an index in a background thread ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUND_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUND_H

#include "Context.h"
#include "FSProvider.h"
#include "GlobalCompilationDatabase.h"
#include "Threading.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/Serialization.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/Threading.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
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

  // The factory provides storage for each CDB.
  // It keeps ownership of the storage instances, and should manage caching
  // itself. Factory must be threadsafe and never returns nullptr.
  using Factory =
      llvm::unique_function<BackgroundIndexStorage *(llvm::StringRef)>;

  // Creates an Index Storage that saves shards into disk. Index storage uses
  // CDBDirectory + ".clangd/index/" as the folder to save shards.
  static Factory createDiskBackedStorageFactory();
};

// Builds an in-memory index by by running the static indexer action over
// all commands in a compilation database. Indexing happens in the background.
// FIXME: it should also persist its state on disk for fast start.
// FIXME: it should watch for changes to files on disk.
class BackgroundIndex : public SwapIndex {
public:
  /// If BuildIndexPeriodMs is greater than 0, the symbol index will only be
  /// rebuilt periodically (one per \p BuildIndexPeriodMs); otherwise, index is
  /// rebuilt for each indexed file.
  BackgroundIndex(
      Context BackgroundContext, const FileSystemProvider &,
      const GlobalCompilationDatabase &CDB,
      BackgroundIndexStorage::Factory IndexStorageFactory,
      size_t BuildIndexPeriodMs = 0,
      size_t ThreadPoolSize = llvm::heavyweight_hardware_concurrency());
  ~BackgroundIndex(); // Blocks while the current task finishes.

  // Enqueue translation units for indexing.
  // The indexing happens in a background thread, so the symbols will be
  // available sometime later.
  void enqueue(const std::vector<std::string> &ChangedFiles);

  // Cause background threads to stop after ther current task, any remaining
  // tasks will be discarded.
  void stop();

  // Wait until the queue is empty, to allow deterministic testing.
  LLVM_NODISCARD bool
  blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds = 10);

  // Disables thread priority lowering in background index to make sure it can
  // progress on loaded systems. Only affects tasks that run after the call.
  static void preventThreadStarvationInTests();

private:
  /// Given index results from a TU, only update symbols coming from files with
  /// different digests than \p DigestsSnapshot. Also stores new index
  /// information on IndexStorage.
  void update(llvm::StringRef MainFile, IndexFileIn Index,
              const llvm::StringMap<FileDigest> &DigestsSnapshot,
              BackgroundIndexStorage *IndexStorage);

  // configuration
  const FileSystemProvider &FSProvider;
  const GlobalCompilationDatabase &CDB;
  Context BackgroundContext;

  // index state
  llvm::Error index(tooling::CompileCommand,
                    BackgroundIndexStorage *IndexStorage);
  void buildIndex(); // Rebuild index periodically every BuildIndexPeriodMs.
  const size_t BuildIndexPeriodMs;
  std::atomic<bool> SymbolsUpdatedSinceLastIndex;
  std::mutex IndexMu;
  std::condition_variable IndexCV;

  FileSymbols IndexedSymbols;
  llvm::StringMap<FileDigest> IndexedFileDigests; // Key is absolute file path.
  std::mutex DigestsMu;

  BackgroundIndexStorage::Factory IndexStorageFactory;
  struct Source {
    std::string Path;
    bool NeedsReIndexing;
    Source(llvm::StringRef Path, bool NeedsReIndexing)
        : Path(Path), NeedsReIndexing(NeedsReIndexing) {}
  };
  // Loads the shards for a single TU and all of its dependencies. Returns the
  // list of sources and whether they need to be re-indexed.
  std::vector<Source> loadShard(const tooling::CompileCommand &Cmd,
                                BackgroundIndexStorage *IndexStorage,
                                llvm::StringSet<> &LoadedShards);
  // Tries to load shards for the ChangedFiles.
  std::vector<std::pair<tooling::CompileCommand, BackgroundIndexStorage *>>
  loadShards(std::vector<std::string> ChangedFiles);
  void enqueue(tooling::CompileCommand Cmd, BackgroundIndexStorage *Storage);

  // queue management
  using Task = std::function<void()>;
  void run(); // Main loop executed by Thread. Runs tasks from Queue.
  void enqueueTask(Task T, llvm::ThreadPriority Prioirty);
  void enqueueLocked(tooling::CompileCommand Cmd,
                     BackgroundIndexStorage *IndexStorage);
  std::mutex QueueMu;
  unsigned NumActiveTasks = 0; // Only idle when queue is empty *and* no tasks.
  std::condition_variable QueueCV;
  bool ShouldStop = false;
  std::deque<std::pair<Task, llvm::ThreadPriority>> Queue;
  AsyncTaskRunner ThreadPool;
  GlobalCompilationDatabase::CommandChanged::Subscription CommandsChanged;
};

} // namespace clangd
} // namespace clang

#endif
