//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/InProcessModuleCache.h"

#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/Support/AdvisoryLock.h"

#include <mutex>

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {
class ReaderWriterLock : public llvm::AdvisoryLock {
  // TODO: Consider using std::atomic::{wait,notify_all} when we move to C++20.
  std::unique_lock<std::shared_mutex> OwningLock;

public:
  ReaderWriterLock(std::shared_mutex &Mutex)
      : OwningLock(Mutex, std::defer_lock) {}

  Expected<bool> tryLock() override { return OwningLock.try_lock(); }

  llvm::WaitForUnlockResult
  waitForUnlockFor(std::chrono::seconds MaxSeconds) override {
    assert(!OwningLock);
    // We do not respect the timeout here. It's very generous for implicit
    // modules, so we'd typically only reach it if the owner crashed (but so did
    // we, since we run in the same process), or encountered deadlock.
    (void)MaxSeconds;
    std::shared_lock<std::shared_mutex> Lock(*OwningLock.mutex());
    return llvm::WaitForUnlockResult::Success;
  }

  std::error_code unsafeMaybeUnlock() override {
    // Unlocking the mutex here would trigger UB and we don't expect this to be
    // actually called when compiling scanning modules due to the no-timeout
    // guarantee above.
    return {};
  }

  ~ReaderWriterLock() override = default;
};

class InProcessModuleCache : public ModuleCache {
  ModuleCacheMutexes &Mutexes;

  // TODO: If we changed the InMemoryModuleCache API and relied on strict
  // context hash, we could probably create more efficient thread-safe
  // implementation of the InMemoryModuleCache such that it doesn't need to be
  // recreated for each translation unit.
  InMemoryModuleCache InMemory;

public:
  InProcessModuleCache(ModuleCacheMutexes &Mutexes) : Mutexes(Mutexes) {}

  void prepareForGetLock(StringRef Filename) override {}

  std::unique_ptr<llvm::AdvisoryLock> getLock(StringRef Filename) override {
    auto &Mtx = [&]() -> std::shared_mutex & {
      std::lock_guard<std::mutex> Lock(Mutexes.Mutex);
      auto &Mutex = Mutexes.Map[Filename];
      if (!Mutex)
        Mutex = std::make_unique<std::shared_mutex>();
      return *Mutex;
    }();
    return std::make_unique<ReaderWriterLock>(Mtx);
  }

  InMemoryModuleCache &getInMemoryModuleCache() override { return InMemory; }
  const InMemoryModuleCache &getInMemoryModuleCache() const override {
    return InMemory;
  }
};
} // namespace

IntrusiveRefCntPtr<ModuleCache>
dependencies::makeInProcessModuleCache(ModuleCacheMutexes &Mutexes) {
  return llvm::makeIntrusiveRefCnt<InProcessModuleCache>(Mutexes);
}
