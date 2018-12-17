//===--- Background.h - Build an index in a background thread ----*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include <condition_variable>
#include <deque>
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
  // CDBDirectory + ".clangd-index/" as the folder to save shards.
  static Factory createDiskBackedStorageFactory();
};

// Builds an in-memory index by by running the static indexer action over
// all commands in a compilation database. Indexing happens in the background.
// FIXME: it should also persist its state on disk for fast start.
// FIXME: it should watch for changes to files on disk.
class BackgroundIndex : public SwapIndex {
public:
  // FIXME: resource-dir injection should be hoisted somewhere common.
  BackgroundIndex(Context BackgroundContext, llvm::StringRef ResourceDir,
                  const FileSystemProvider &,
                  const GlobalCompilationDatabase &CDB,
                  BackgroundIndexStorage::Factory IndexStorageFactory,
                  size_t ThreadPoolSize = llvm::hardware_concurrency());
  ~BackgroundIndex(); // Blocks while the current task finishes.

  // Enqueue translation units for indexing.
  // The indexing happens in a background thread, so the symbols will be
  // available sometime later.
  void enqueue(const std::vector<std::string> &ChangedFiles);
  void enqueue(const std::string &File);

  // Cause background threads to stop after ther current task, any remaining
  // tasks will be discarded.
  void stop();

  // Wait until the queue is empty, to allow deterministic testing.
  LLVM_NODISCARD bool
  blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds = 10);

private:
  /// Given index results from a TU, only update files in \p FilesToUpdate.
  /// Also stores new index information on IndexStorage.
  void update(llvm::StringRef MainFile, IndexFileIn Index,
              const llvm::StringMap<FileDigest> &FilesToUpdate,
              BackgroundIndexStorage *IndexStorage);

  // configuration
  std::string ResourceDir;
  const FileSystemProvider &FSProvider;
  const GlobalCompilationDatabase &CDB;
  Context BackgroundContext;

  // index state
  llvm::Error index(tooling::CompileCommand,
                    BackgroundIndexStorage *IndexStorage);

  FileSymbols IndexedSymbols;
  llvm::StringMap<FileDigest> IndexedFileDigests; // Key is absolute file path.
  std::mutex DigestsMu;

  BackgroundIndexStorage::Factory IndexStorageFactory;

  // queue management
  using Task = std::function<void()>;
  void run(); // Main loop executed by Thread. Runs tasks from Queue.
  void enqueueTask(Task T, ThreadPriority Prioirty);
  void enqueueLocked(tooling::CompileCommand Cmd,
                     BackgroundIndexStorage *IndexStorage);
  std::mutex QueueMu;
  unsigned NumActiveTasks = 0; // Only idle when queue is empty *and* no tasks.
  std::condition_variable QueueCV;
  bool ShouldStop = false;
  std::deque<std::pair<Task, ThreadPriority>> Queue;
  std::vector<std::thread> ThreadPool; // FIXME: Abstract this away.
  GlobalCompilationDatabase::CommandChanged::Subscription CommandsChanged;
};

} // namespace clangd
} // namespace clang

#endif
