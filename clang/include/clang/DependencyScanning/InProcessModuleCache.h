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

  ~ModuleCacheEntries();

  /// Flushes all PCMs built in-process that haven't been flushed yet to disk.
  void flush();

  /// Starts a background thread that flushes PCMs built in-process to disk as
  /// soon as they're written, replacing their in-memory buffers with
  /// memory-mapped files.
  void startFlushing(AtomicLineLogger &Logger);

  /// Waits for the background thread to flush all written PCMs and stops it.
  /// PCMs that failed to be written are kept in memory only.
  void stopFlushing();

  /// Hands \p MapEntry, which was just written, to the background thread.
  void addPending(ModuleCacheMapEntry &MapEntry);

private:
  /// The body of the background thread, see \c startFlushing().
  void flushInBackground(AtomicLineLogger &Logger);

  /// Flushes single entry, see \c startFlushing().
  void flushEntry(ModuleCacheMapEntry &MapEntry, AtomicLineLogger &Logger);

  /// Entries waiting to be flushed by the background thread. These point into
  /// \c Map, whose entries have stable addresses.
  std::mutex PendingMutex;
  std::vector<ModuleCacheMapEntry *> Pending;
  /// Notifies the background thread about new pending entries or stopping.
  std::condition_variable PendingCondVar;
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
