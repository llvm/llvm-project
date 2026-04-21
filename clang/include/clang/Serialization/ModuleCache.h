//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_MODULECACHE_H
#define LLVM_CLANG_SERIALIZATION_MODULECACHE_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem/UniqueID.h"

#include <ctime>
#include <memory>
#include <sys/types.h>
#include <system_error>

namespace llvm {
class AdvisoryLock;
class MemoryBuffer;
class MemoryBufferRef;
} // namespace llvm

namespace clang {
class InMemoryModuleCache;

/// The address of an instance of this class represents the identity of a module
/// cache directory.
class ModuleCacheDirectory {};

/// The module cache used for compiling modules implicitly. This centralizes the
/// operations the compiler might want to perform on the cache.
class ModuleCache {
  /// Mapping from a path to the module cache directory identity.
  llvm::StringMap<const ModuleCacheDirectory *> ByPath;

  /// Mapping from the filesystem entity to the module cache directory identity.
  llvm::DenseMap<llvm::sys::fs::UniqueID, std::unique_ptr<ModuleCacheDirectory>>
      ByUID;

public:
  /// Returns an opaque pointer representing the module cache directory. This
  /// returns the same pointer regardless of the path spelling, as long as it
  /// resolves to the same file system entity. This also resolves links in the
  /// path. This may return nullptr if the module cache does not exist.
  virtual const ModuleCacheDirectory *getDirectoryPtr(StringRef Path);

  /// Returns lock for the given module file. The lock is initially unlocked.
  virtual std::unique_ptr<llvm::AdvisoryLock>
  getLock(StringRef ModuleFilename) = 0;

  // TODO: Abstract away timestamps with isUpToDate() and markUpToDate().
  // TODO: Consider exposing a "validation lock" API to prevent multiple clients
  // concurrently noticing an out-of-date module file and validating its inputs.

  /// Returns the timestamp denoting the last time inputs of the module file
  /// were validated.
  virtual std::time_t getModuleTimestamp(StringRef ModuleFilename) = 0;

  /// Updates the timestamp denoting the last time inputs of the module file
  /// were validated.
  virtual void updateModuleTimestamp(StringRef ModuleFilename) = 0;

  /// Prune module files that haven't been accessed in a long time.
  virtual void maybePrune(StringRef Path, time_t PruneInterval,
                          time_t PruneAfter) = 0;

  /// Returns this process's view of the module cache.
  virtual InMemoryModuleCache &getInMemoryModuleCache() = 0;
  virtual const InMemoryModuleCache &getInMemoryModuleCache() const = 0;

  /// Write the PCM contents to the given path in the module cache.
  virtual std::error_code write(StringRef Path, llvm::MemoryBufferRef Buffer,
                                off_t &Size, time_t &ModTime) = 0;

  virtual Expected<std::unique_ptr<llvm::MemoryBuffer>>
  read(StringRef FileName, off_t &Size, time_t &ModTime) = 0;

  virtual ~ModuleCache() = default;
};

/// Creates new \c ModuleCache backed by a file system directory that may be
/// operated on by multiple processes. This instance must be used across all
/// \c CompilerInstance instances participating in building modules for single
/// translation unit in order to share the same \c InMemoryModuleCache.
std::shared_ptr<ModuleCache> createCrossProcessModuleCache();

/// Shared implementation of `ModuleCache::maybePrune()`.
void maybePruneImpl(StringRef Path, time_t PruneInterval, time_t PruneAfter,
                    bool PruneTopLevel = false);

/// Shared implementation of `ModuleCache::write()`.
std::error_code writeImpl(StringRef Path, llvm::MemoryBufferRef Buffer,
                          off_t &Size, time_t &ModTime);

/// Shared implementation of `ModuleCache::read()`.
Expected<std::unique_ptr<llvm::MemoryBuffer>>
readImpl(StringRef FileName, off_t &Size, time_t &ModTime);
} // namespace clang

#endif
