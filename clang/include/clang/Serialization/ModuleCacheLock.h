//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_MODULECACHELOCK_H
#define LLVM_CLANG_SERIALIZATION_MODULECACHELOCK_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/LockFileManager.h"

namespace clang {
enum class LockResult { Owned, Shared, Error };
enum class WaitForUnlockResult { Success, OwnerDied, Timeout };

class ModuleCacheLockManager {
public:
  virtual operator LockResult() const = 0;
  virtual WaitForUnlockResult waitForUnlock() = 0;
  virtual void unsafeRemoveLock() = 0;
  virtual std::string getErrorMessage() const = 0;
  virtual ~ModuleCacheLockManager() = default;
};

class ModuleCacheLock {
public:
  virtual void prepareLock(StringRef ModuleFilename) = 0;
  virtual std::unique_ptr<ModuleCacheLockManager>
  tryLock(StringRef ModuleFilename) = 0;
  virtual ~ModuleCacheLock() = default;
};

std::shared_ptr<ModuleCacheLock> getModuleCacheFileLock();
} // namespace clang

#endif
