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
#include "llvm/Support/Chrono.h"

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
  ModuleCacheEntries &Entries;

  // TODO: If we changed the InMemoryModuleCache API and relied on strict
  // context hash, we could probably create more efficient thread-safe
  // implementation of the InMemoryModuleCache such that it doesn't need to be
  // recreated for each translation unit.
  InMemoryModuleCache InMemory;

public:
  InProcessModuleCache(ModuleCacheEntries &Entries) : Entries(Entries) {}

  void prepareForGetLock(StringRef Filename) override {}

  std::unique_ptr<llvm::AdvisoryLock> getLock(StringRef Filename) override {
    auto &CompilationMutex = [&]() -> std::shared_mutex & {
      std::lock_guard<std::mutex> Lock(Entries.Mutex);
      auto &Entry = Entries.Map[Filename];
      if (!Entry)
        Entry = std::make_unique<ModuleCacheEntry>();
      return Entry->CompilationMutex;
    }();
    return std::make_unique<ReaderWriterLock>(CompilationMutex);
  }

  std::time_t getModuleTimestamp(StringRef Filename) override {
    auto &Timestamp = [&]() -> std::atomic<std::time_t> & {
      std::lock_guard<std::mutex> Lock(Entries.Mutex);
      auto &Entry = Entries.Map[Filename];
      if (!Entry)
        Entry = std::make_unique<ModuleCacheEntry>();
      return Entry->Timestamp;
    }();

    return Timestamp.load();
  }

  void updateModuleTimestamp(StringRef Filename) override {
    // Note: This essentially replaces FS contention with mutex contention.
    auto &Timestamp = [&]() -> std::atomic<std::time_t> & {
      std::lock_guard<std::mutex> Lock(Entries.Mutex);
      auto &Entry = Entries.Map[Filename];
      if (!Entry)
        Entry = std::make_unique<ModuleCacheEntry>();
      return Entry->Timestamp;
    }();

    Timestamp.store(llvm::sys::toTimeT(std::chrono::system_clock::now()));
  }

  void maybePrune(StringRef Path, time_t PruneInterval,
                  time_t PruneAfter) override {
    // FIXME: This only needs to be ran once per build, not in every
    // compilation. Call it once per service.
    maybePruneImpl(Path, PruneInterval, PruneAfter);
  }

  InMemoryModuleCache &getInMemoryModuleCache() override { return InMemory; }
  const InMemoryModuleCache &getInMemoryModuleCache() const override {
    return InMemory;
  }
};
} // namespace

IntrusiveRefCntPtr<ModuleCache>
dependencies::makeInProcessModuleCache(ModuleCacheEntries &Entries) {
  return llvm::makeIntrusiveRefCnt<InProcessModuleCache>(Entries);
}
