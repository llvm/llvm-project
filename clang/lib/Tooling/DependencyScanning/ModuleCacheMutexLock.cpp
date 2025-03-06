//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/ModuleCacheMutexLock.h"

#include <mutex>

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {
struct ModuleCacheMutexLockManager : ModuleCacheLockManager {
  std::unique_lock<std::shared_mutex> OwningLock;

  ModuleCacheMutexLockManager(std::shared_mutex &Mutex)
      : OwningLock(Mutex, std::try_to_lock_t{}) {}

  operator LockResult() const override {
    return OwningLock ? LockResult::Owned : LockResult::Shared;
  }

  WaitForUnlockResult waitForUnlock() override {
    assert(!OwningLock);
    std::shared_lock Lock(*OwningLock.mutex());
    return WaitForUnlockResult::Success;
  }

  void unsafeRemoveLock() override {
    llvm_unreachable("ModuleCacheMutexLockManager cannot remove locks");
  }

  std::string getErrorMessage() const override {
    llvm_unreachable("ModuleCacheMutexLockManager cannot fail");
  }
};

struct ModuleCacheMutexLock : ModuleCacheLock {
  ModuleCacheMutexes &Mutexes;

  ModuleCacheMutexLock(ModuleCacheMutexes &Mutexes) : Mutexes(Mutexes) {}

  void prepareLock(StringRef Filename) override {}

  std::unique_ptr<ModuleCacheLockManager> tryLock(StringRef Filename) override {
    auto &Mutex = [&]() -> std::shared_mutex & {
      std::lock_guard Lock(Mutexes.Mutex);
      auto &Mutex = Mutexes.Map[Filename];
      if (!Mutex)
        Mutex = std::make_unique<std::shared_mutex>();
      return *Mutex;
    }();
    return std::make_unique<ModuleCacheMutexLockManager>(Mutex);
  }
};
} // namespace

std::shared_ptr<ModuleCacheLock>
dependencies::getModuleCacheMutexLock(ModuleCacheMutexes &Mutexes) {
  return std::make_shared<ModuleCacheMutexLock>(Mutexes);
}
