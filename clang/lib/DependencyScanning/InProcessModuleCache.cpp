//===- InProcessModuleCache.cpp - Implicit Module Cache ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/InProcessModuleCache.h"

#include "clang/Basic/AtomicLineLogger.h"
#include "clang/Serialization/InMemoryModuleCache.h"
#include "clang/Serialization/ModuleCache.h"
#include "llvm/Support/AdvisoryLock.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace dependencies;

void ModuleCacheEntries::addInvalidatedPath(StringRef Path) {
#ifndef NDEBUG
  SmallString<256> Canonical(Path);
  assert(llvm::sys::path::is_absolute(Path) &&
         !llvm::sys::path::remove_dots(Canonical, /*remove_dot_dot=*/true) &&
         "invalidated paths must be spelled like ModuleDeps::DirectoryDeps");
#endif
  {
    std::lock_guard<std::mutex> Lock(InvalidatedPathsMutex);
    InvalidatedPaths.insert(Path);
  }
  AnyInvalidatedPaths.store(true, std::memory_order_release);
}

std::optional<bool>
ModuleCacheEntries::isDirectoryInvalidated(StringRef Directory) const {
  if (!ValidateAgainstInvalidatedPaths)
    return std::nullopt;
  if (!AnyInvalidatedPaths.load(std::memory_order_acquire))
    return false;
  std::lock_guard<std::mutex> Lock(InvalidatedPathsMutex);
  return InvalidatedPaths.contains(Directory);
}

ModuleCacheEntries::~ModuleCacheEntries() { stopFlushing(); }

void ModuleCacheEntries::flush() {
  auto BypassSandbox = llvm::sys::sandbox::scopedDisable();
  for (auto &[Path, Entry] : Map) {
    if (Entry->State == ModuleCacheEntry::S_Written) {
      assert(Entry->Buffer && "Wrote PCM with no contents");
      // Note: We could propagate Entry->ModTime to the on-disk file, but
      // implicitly-built modules (unlike explicitly-built modules) don't use
      // that metadata to refer to imports, rendering this unnecessary.
      off_t Size;
      time_t ModTime;
      // Best-effort: ignore errors (e.g. read-only cache directory).
      (void)writeImpl(Path, *Entry->Buffer, Size, ModTime,
                      /*ReadBack=*/false);
    }
  }
}

void ModuleCacheEntries::startFlushing(AtomicLineLogger &Logger) {
  assert(!Flusher.joinable() && "Already flushing in the background");
  Flusher = std::thread([this, &Logger] { flushInBackground(Logger); });
}

void ModuleCacheEntries::stopFlushing() {
  if (!Flusher.joinable())
    return;
  {
    std::lock_guard<std::mutex> Lock(PendingMutex);
    StopFlushing = true;
  }
  PendingCondVar.notify_one();
  Flusher.join();
}

void ModuleCacheEntries::addPending(ModuleCacheMapEntry &MapEntry) {
  // Note: The background thread only gets started or stopped while no PCMs
  // are being written.
  if (!Flusher.joinable())
    return;
  {
    std::lock_guard<std::mutex> Lock(PendingMutex);
    Pending.push_back(&MapEntry);
  }
  PendingCondVar.notify_one();
}

void ModuleCacheEntries::flushInBackground(AtomicLineLogger &Logger) {
  // This is a compiler-internal output, let's bypass the sandbox.
  auto BypassSandbox = llvm::sys::sandbox::scopedDisable();
  std::unique_lock<std::mutex> Lock(PendingMutex);
  while (true) {
    PendingCondVar.wait(Lock, [&] { return StopFlushing || !Pending.empty(); });
    // Flush everything that's been written before stopping.
    if (Pending.empty())
      return;
    std::vector<ModuleCacheMapEntry *> Batch;
    std::swap(Batch, Pending);
    Lock.unlock();
    for (ModuleCacheMapEntry *MapEntry : Batch)
      flushEntry(*MapEntry, Logger);
    Lock.lock();
  }
}

void ModuleCacheEntries::flushEntry(ModuleCacheMapEntry &MapEntry,
                                    AtomicLineLogger &Logger) {
  StringRef Path = MapEntry.getKey();
  ModuleCacheEntry &Entry = *MapEntry.second;
  std::shared_ptr<const llvm::MemoryBuffer> InMemory;
  {
    std::lock_guard<std::mutex> Lock(Entry.Mutex);
    if (Entry.State != ModuleCacheEntry::S_Written)
      return;
    InMemory = Entry.Buffer;
  }

  // The contents of an entry in the S_Written state don't change, so we can
  // write them without holding the lock. This way, importers of the freshly
  // built PCM don't need to wait for the IO.
  off_t Size;
  time_t ModTime;
  auto Written = writeImpl(Path, *InMemory, Size, ModTime, /*ReadBack=*/true);
  if (!Written) {
    // Best-effort: keep the buffer in memory and let flush() try again.
    Logger.log() << "pcm_flush_failed: " << Path;
    return;
  }
  Logger.log() << "pcm_flush: " << Path;

  std::lock_guard<std::mutex> Lock(Entry.Mutex);
  // Note: We keep Entry.ModTime, since that's what readers have already seen.
  // Small files don't get memory-mapped, in which case we keep the original
  // buffer to avoid churn. Readers keep the original buffer alive on their own.
  if ((*Written)->getBufferKind() == llvm::MemoryBuffer::MemoryBuffer_MMap)
    Entry.Buffer = std::move(*Written);
  Entry.State = ModuleCacheEntry::S_Flushed;
}

namespace {
/// A buffer that keeps the contents of a \c ModuleCacheEntry alive.
class SharedMemoryBuffer : public llvm::MemoryBuffer {
  std::shared_ptr<const llvm::MemoryBuffer> Underlying;

public:
  SharedMemoryBuffer(std::shared_ptr<const llvm::MemoryBuffer> Underlying)
      : Underlying(std::move(Underlying)) {
    init(this->Underlying->getBufferStart(), this->Underlying->getBufferEnd(),
         /*RequiresNullTerminator=*/false);
  }

  StringRef getBufferIdentifier() const override {
    return Underlying->getBufferIdentifier();
  }

  BufferKind getBufferKind() const override {
    return Underlying->getBufferKind();
  }
};

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

  ModuleCacheMapEntry &getOrCreateMapEntry(StringRef Filename) {
    std::lock_guard<std::mutex> Lock(Entries.Mutex);
    auto &MapEntry = *Entries.Map.try_emplace(Filename).first;
    if (!MapEntry.second)
      MapEntry.second = std::make_unique<ModuleCacheEntry>();
    return MapEntry;
  }

  ModuleCacheEntry &getOrCreateEntry(StringRef Filename) {
    return *getOrCreateMapEntry(Filename).second;
  }

public:
  InProcessModuleCache(ModuleCacheEntries &Entries, AtomicLineLogger &Logger)
      : ModuleCache(Logger), Entries(Entries), InMemory(Logger) {}

  std::unique_ptr<llvm::AdvisoryLock> getLock(StringRef Filename) override {
    auto &Entry = getOrCreateEntry(Filename);
    return std::make_unique<ReaderWriterLock>(Entry);
  }

  std::time_t getModuleTimestamp(StringRef Filename) override {
    auto &Timestamp = getOrCreateEntry(Filename).Timestamp;

    Logger.log() << "timestamp_read: " << Filename;
    return Timestamp.load();
  }

  void updateModuleTimestamp(StringRef Filename) override {
    // Note: This essentially replaces FS contention with mutex contention.
    auto &Timestamp = getOrCreateEntry(Filename).Timestamp;

    Logger.log() << "timestamp_write: " << Filename;
    Timestamp.store(llvm::sys::toTimeT(std::chrono::system_clock::now()));
  }

  std::optional<bool> isDirectoryInvalidated(StringRef Directory) override {
    return Entries.isDirectoryInvalidated(Directory);
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

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  write(StringRef Path, std::unique_ptr<llvm::MemoryBuffer> Buffer, off_t &Size,
        time_t &ModTime) override {
    ModuleCacheMapEntry &MapEntry = getOrCreateMapEntry(Path);
    ModuleCacheEntry &Entry = *MapEntry.second;
    std::lock_guard<std::mutex> Lock(Entry.Mutex);
    Logger.log() << "pcm_write: " << Path;
    if (Entry.State == ModuleCacheEntry::S_Written ||
        Entry.State == ModuleCacheEntry::S_Flushed) {
      assert(Entry.Buffer && "Wrote PCM with no contents");
      assert(Entry.Buffer->getBuffer() == Buffer->getBuffer() &&
             "Wrote the same PCM with different contents");
      Size = Entry.Buffer->getBufferSize();
      ModTime = Entry.ModTime;
      return std::make_unique<SharedMemoryBuffer>(Entry.Buffer);
    }
    // Readers of the previous buffer (if any) keep it alive on their own.
    Entry.Buffer = std::move(Buffer);
    Entry.ModTime = llvm::sys::toTimeT(std::chrono::system_clock::now());
    Entry.State = ModuleCacheEntry::S_Written;
    Entries.addPending(MapEntry);
    Size = Entry.Buffer->getBufferSize();
    ModTime = Entry.ModTime;
    return std::make_unique<SharedMemoryBuffer>(Entry.Buffer);
  }

  Expected<std::unique_ptr<llvm::MemoryBuffer>>
  read(StringRef FileName, off_t &Size, time_t &ModTime) override {
    Logger.log() << "pcm_read_disk: " << FileName;
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
    return std::make_unique<SharedMemoryBuffer>(Entry.Buffer);
  }
};
} // namespace

std::shared_ptr<ModuleCache>
dependencies::makeInProcessModuleCache(ModuleCacheEntries &Entries,
                                       AtomicLineLogger &Logger) {
  return std::make_shared<InProcessModuleCache>(Entries, Logger);
}
