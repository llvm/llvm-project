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
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace llvm {
class AdvisoryLock;
} // namespace llvm

namespace clang {
class InMemoryModuleCache;

/// The module cache used by implicitly-built modules.
class ModuleCache : public RefCountedBase<ModuleCache> {
public:
  /// May perform any work that only needs to be performed once for multiple
  /// calls \c getLock() with the same module filename.
  virtual void prepareForGetLock(StringRef ModuleFilename) = 0;

  /// Returns lock for the given module file. The lock is initially unlocked.
  virtual std::unique_ptr<llvm::AdvisoryLock>
  getLock(StringRef ModuleFilename) = 0;

  /// Returns this process's view of the module cache.
  virtual InMemoryModuleCache &getInMemoryModuleCache() = 0;
  virtual const InMemoryModuleCache &getInMemoryModuleCache() const = 0;

  // TODO: Virtualize writing/reading PCM files, timestamp files, etc.

  virtual ~ModuleCache() = default;
};

IntrusiveRefCntPtr<ModuleCache> getCrossProcessModuleCache();
} // namespace clang

#endif
