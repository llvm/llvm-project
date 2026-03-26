//===- InProcessModuleCache.cpp - Implicit Module Cache ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/InProcessModuleCache.h"

#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/Support/AdvisoryLock.h"
#include "llvm/Support/Chrono.h"

using namespace clang;
using namespace dependencies;

namespace {
class ReaderWriterLock : public llvm::AdvisoryLock {
  ModuleCacheEntry &Entry;
  std::optional<unsigned> OwnedGeneration;

public:
  ReaderWriterLock(ModuleCacheEntry &Entry) : Entry(Entry) {}

  Expected<bool> tryLock() override {
    std::lock_guard<std::mutex> Lock(Entry.Mutex);
    if (Entry.Locked)
      return false;
    Entry.Locked = true;
    OwnedGeneration = Entry.Generation;
    return true;
  }

  llvm::WaitForUnlockResult
  waitForUnlockFor(std::chrono::seconds MaxSeconds) override {
    assert(!OwnedGeneration);
    std::unique_lock<std::mutex> Lock(Entry.Mutex);
    unsigned CurrentGeneration = Entry.Generation;
    bool Success = Entry.CondVar.wait_for(Lock, MaxSeconds, [&] {
      // We check not only Locked, but also Generation to break the wait in case
      // of unsafeUnlock() and successful tryLock().
      return !Entry.Locked || Entry.Generation != CurrentGeneration;
    });
    return Success ? llvm::WaitForUnlockResult::Success
                   : llvm::WaitForUnlockResult::Timeout;
  }

  std::error_code unsafeUnlock() override {
    {
      std::lock_guard<std::mutex> Lock(Entry.Mutex);
      Entry.Generation += 1;
      Entry.Locked = false;
    }
    Entry.CondVar.notify_all();
    return {};
  }

  ~ReaderWriterLock() override {
    if (OwnedGeneration) {
      {
        std::lock_guard<std::mutex> Lock(Entry.Mutex);
        // Avoid stomping over the state managed by someone else after
        // unsafeUnlock() and successful tryLock().
        if (*OwnedGeneration == Entry.Generation)
          Entry.Locked = false;
      }
      Entry.CondVar.notify_all();
    }
  }
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
    auto &Entry = [&]() -> ModuleCacheEntry & {
      std::lock_guard<std::mutex> Lock(Entries.Mutex);
      auto &Entry = Entries.Map[Filename];
      if (!Entry)
        Entry = std::make_unique<ModuleCacheEntry>();
      return *Entry;
    }();
    return std::make_unique<ReaderWriterLock>(Entry);
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

std::shared_ptr<ModuleCache>
dependencies::makeInProcessModuleCache(ModuleCacheEntries &Entries) {
  return std::make_shared<InProcessModuleCache>(Entries);
}
