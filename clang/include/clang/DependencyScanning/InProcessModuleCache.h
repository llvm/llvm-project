//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_INPROCESSMODULECACHE_H
#define LLVM_CLANG_DEPENDENCYSCANNING_INPROCESSMODULECACHE_H

#include "clang/Basic/AtomicLineLogger.h"
#include "clang/Serialization/ModuleCache.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace clang {
namespace dependencies {

struct ModuleCacheEntry {
  std::mutex Mutex;
  std::condition_variable CondVar;
  bool Locked = false;
  unsigned Generation = 0;

  std::atomic<std::time_t> Timestamp = 0;

  enum {
    S_Unknown,
    S_Read,
    /// Written to the module cache, but only kept in memory.
    S_Written,
    /// Written to the module cache and flushed to disk.
    S_Flushed,
  } State = S_Unknown;
  /// The buffer we've read from disk or written to the module cache, if any.
  /// Buffers returned from \c ModuleCache::read() share ownership of this, so
  /// it can be replaced while readers still reference the previous contents.
  std::shared_ptr<const llvm::MemoryBuffer> Buffer;
  /// The modification time of the entry.
  time_t ModTime = 0;
};

using ModuleCacheMapEntry =
    llvm::StringMapEntry<std::unique_ptr<ModuleCacheEntry>>;

struct ModuleCacheEntries {
  std::mutex Mutex;
  llvm::StringMap<std::unique_ptr<ModuleCacheEntry>> Map;

  bool ValidateAgainstInvalidatedPaths = false;

  void addInvalidatedPath(StringRef Path);
  std::optional<bool> isDirectoryInvalidated(StringRef Directory) const;

  /// Adds \p MapEntry into the queue of entries to be flushed.
  void scheduleFlush(ModuleCacheMapEntry &MapEntry);

  /// Starts a background thread that flushes PCMs built in-process to disk as
  /// soon as they're written, replacing their in-memory buffers with
  /// memory-mapped files.
  void startBackgroundFlushing(AtomicLineLogger &Logger);

  /// Stops the background thread as soon as it finishes flushing the current
  /// PCM. PCMs it didn't get to are left for \c flushLeftovers().
  void stopBackgroundFlushing();

  /// Flushes all PCMs built in-process that haven't been flushed yet.
  void flushLeftovers(AtomicLineLogger &Logger);

  ~ModuleCacheEntries();

private:
  /// Flushes a single cache entry.
  static void flushEntry(StringRef Path, ModuleCacheEntry &Entry,
                         AtomicLineLogger &Logger, bool ReadBack);

  /// Entries waiting to be flushed. These point into \c Map, whose entries have
  /// stable addresses.
  std::mutex FlushMutex;
  std::deque<ModuleCacheMapEntry *> FlushEntries;
  /// Notifies the background thread about new pending entries or stopping.
  std::condition_variable FlushCondVar;
  bool StopFlushing = false;
  std::thread Flusher;

  mutable std::mutex InvalidatedPathsMutex;
  llvm::StringSet<> InvalidatedPaths;
  std::atomic<bool> AnyInvalidatedPaths = false;
};

std::shared_ptr<ModuleCache>
makeInProcessModuleCache(ModuleCacheEntries &Entries, AtomicLineLogger &Logger);

} // namespace dependencies
} // namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_INPROCESSMODULECACHE_H
