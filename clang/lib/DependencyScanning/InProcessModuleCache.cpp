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
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace dependencies;

void ModuleCacheEntries::flush() {
  auto BypassSandbox = llvm::sys::sandbox::scopedDisable();
  for (auto &[Path, Entry] : Map) {
    if (Entry->State == ModuleCacheEntry::S_Written) {
      // Note: We could propagate Entry->ModTime to the on-disk file, but
      // implicitly-built modules (unlike explicitly-built modules) don't use
      // that metadata to refer to imports, rendering this unnecessary.
      off_t Size;
      time_t ModTime;
      // Best-effort: ignore errors (e.g. read-only cache directory).
      (void)writeImpl(Path, Entry->Buffer->getMemBufferRef(), Size, ModTime);
    }
  }
}

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

  ModuleCacheEntry &getOrCreateEntry(StringRef Filename) {
    std::lock_guard<std::mutex> Lock(Entries.Mutex);
    auto &Entry = Entries.Map[Filename];
    if (!Entry)
      Entry = std::make_unique<ModuleCacheEntry>();
    return *Entry;
  }

public:
  InProcessModuleCache(ModuleCacheEntries &Entries) : Entries(Entries) {}

  std::unique_ptr<llvm::AdvisoryLock> getLock(StringRef Filename) override {
    auto &Entry = getOrCreateEntry(Filename);
    return std::make_unique<ReaderWriterLock>(Entry);
  }

  std::time_t getModuleTimestamp(StringRef Filename) override {
    auto &Timestamp = getOrCreateEntry(Filename).Timestamp;

    return Timestamp.load();
  }

  void updateModuleTimestamp(StringRef Filename) override {
    // Note: This essentially replaces FS contention with mutex contention.
    auto &Timestamp = getOrCreateEntry(Filename).Timestamp;

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

  std::error_code write(StringRef Path, llvm::MemoryBufferRef Buffer,
                        off_t &Size, time_t &ModTime) override {
    ModuleCacheEntry &Entry = getOrCreateEntry(Path);
    std::lock_guard<std::mutex> Lock(Entry.Mutex);
    if (Entry.State == ModuleCacheEntry::S_Written) {
      assert(Entry.Buffer && "Wrote PCM with no contents");
      assert(Entry.Buffer->getBuffer() == Buffer.getBuffer() &&
             "Wrote the same PCM with different contents");
      Size = Entry.Buffer->getBufferSize();
      ModTime = Entry.ModTime;
      return {};
    }
    Entry.Buffer =
        llvm::MemoryBuffer::getMemBufferCopy(Buffer.getBuffer(), Path);
    Entry.ModTime = llvm::sys::toTimeT(std::chrono::system_clock::now());
    Entry.State = ModuleCacheEntry::S_Written;
    Size = Entry.Buffer->getBufferSize();
    ModTime = Entry.ModTime;
    return {};
  }

  Expected<std::unique_ptr<llvm::MemoryBuffer>>
  read(StringRef FileName, off_t &Size, time_t &ModTime) override {
    ModuleCacheEntry &Entry = getOrCreateEntry(FileName);
    std::lock_guard<std::mutex> Lock(Entry.Mutex);
    if (Entry.State == ModuleCacheEntry::S_Unknown) {
      // This is a compiler-internal input/output, let's bypass the sandbox.
      auto BypassSandbox = llvm::sys::sandbox::scopedDisable();
      off_t ReadSize;
      time_t ReadModTime;
      auto ReadBuffer = readImpl(FileName, ReadSize, ReadModTime);
      if (!ReadBuffer)
        return ReadBuffer.takeError();
      Entry.Buffer = std::move(*ReadBuffer);
      Entry.ModTime = ReadModTime;
      Entry.State = ModuleCacheEntry::S_Read;
    }
    Size = Entry.Buffer->getBufferSize();
    ModTime = Entry.ModTime;
    return llvm::MemoryBuffer::getMemBuffer(*Entry.Buffer,
                                            /* RequiresNullTerminator */ false);
  }
};
} // namespace

std::shared_ptr<ModuleCache>
dependencies::makeInProcessModuleCache(ModuleCacheEntries &Entries) {
  return std::make_shared<InProcessModuleCache>(Entries);
}
