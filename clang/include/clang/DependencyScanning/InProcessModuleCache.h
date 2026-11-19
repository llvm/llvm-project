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
    S_Written,
  } State = S_Unknown;
  /// The buffer we've read from disk or written to the module cache, if any.
  /// Buffers returned from \c ModuleCache::read() share ownership of this, so
  /// it can be replaced while readers still reference the previous contents.
  std::shared_ptr<const llvm::MemoryBuffer> Buffer;
  /// The modification time of the entry.
  time_t ModTime = 0;
};

struct ModuleCacheEntries {
  std::mutex Mutex;
  llvm::StringMap<std::unique_ptr<ModuleCacheEntry>> Map;

  bool ValidateAgainstInvalidatedPaths = false;

  void addInvalidatedPath(StringRef Path);
  std::optional<bool> isDirectoryInvalidated(StringRef Directory) const;

  /// Flushes all PCMs built in-process to disk.
  void flush();

private:
  mutable std::mutex InvalidatedPathsMutex;
  llvm::StringSet<> InvalidatedPaths;
  std::atomic<bool> AnyInvalidatedPaths = false;
};

std::shared_ptr<ModuleCache>
makeInProcessModuleCache(ModuleCacheEntries &Entries, AtomicLineLogger &Logger);

} // namespace dependencies
} // namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_INPROCESSMODULECACHE_H
