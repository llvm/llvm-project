//===- DependencyScanningFilesystem.cpp - clang-scan-deps fs --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/Threading.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace dependencies;

llvm::ErrorOr<DependencyScanningWorkerFilesystem::TentativeEntry>
DependencyScanningWorkerFilesystem::readFile(StringRef Filename) {
  // Load the file and its content from the file system.
  auto MaybeFile = getUnderlyingFS().openFileForRead(Filename);
  if (!MaybeFile)
    return MaybeFile.getError();
  auto File = std::move(*MaybeFile);

  auto MaybeStat = File->status();
  if (!MaybeStat)
    return MaybeStat.getError();
  auto Stat = std::move(*MaybeStat);

  auto MaybeBuffer = File->getBuffer(Stat.getName());
  if (!MaybeBuffer)
    return MaybeBuffer.getError();
  auto Buffer = std::move(*MaybeBuffer);

  // If the file size changed between read and stat, pretend it didn't.
  if (Stat.getSize() != Buffer->getBufferSize())
    Stat = llvm::vfs::Status::copyWithNewSize(Stat, Buffer->getBufferSize());

  return TentativeEntry(Stat, std::move(Buffer));
}

EntryRef DependencyScanningWorkerFilesystem::scanForDirectivesIfNecessary(
    const CachedFileSystemEntry &Entry, StringRef Filename, bool Disable) {
  if (Entry.isError() || Entry.isDirectory() || Disable ||
      !shouldScanForDirectives(Filename))
    return EntryRef(Filename, Entry);

  CachedFileContents *Contents = Entry.getCachedContents();
  assert(Contents && "contents not initialized");

  // Double-checked locking.
  if (Contents->DepDirectives.load())
    return EntryRef(Filename, Entry);

  std::lock_guard<std::mutex> GuardLock(Contents->ValueLock);

  // Double-checked locking.
  if (Contents->DepDirectives.load())
    return EntryRef(Filename, Entry);

  SmallVector<dependency_directives_scan::Directive, 64> Directives;
  // Scan the file for preprocessor directives that might affect the
  // dependencies.
  if (scanSourceForDependencyDirectives(Contents->Original->getBuffer(),
                                        Contents->DepDirectiveTokens,
                                        Directives)) {
    Contents->DepDirectiveTokens.clear();
    // FIXME: Propagate the diagnostic if desired by the client.
    Contents->DepDirectives.store(new std::optional<DependencyDirectivesTy>());
    return EntryRef(Filename, Entry);
  }

  // This function performed double-checked locking using `DepDirectives`.
  // Assigning it must be the last thing this function does, otherwise other
  // threads may skip the
  // critical section (`DepDirectives != nullptr`), leading to a data race.
  Contents->DepDirectives.store(
      new std::optional<DependencyDirectivesTy>(std::move(Directives)));
  return EntryRef(Filename, Entry);
}

DependencyScanningFilesystemSharedCache::
    DependencyScanningFilesystemSharedCache() {
  // This heuristic was chosen using a empirical testing on a
  // reasonably high core machine (iMacPro 18 cores / 36 threads). The cache
  // sharding gives a performance edge by reducing the lock contention.
  // FIXME: A better heuristic might also consider the OS to account for
  // the different cost of lock contention on different OSes.
  NumShards =
      std::max(2u, llvm::hardware_concurrency().compute_thread_count() / 4);
  CacheShards = std::make_unique<CacheShard[]>(NumShards);
}

DependencyScanningFilesystemSharedCache::CacheShard &
DependencyScanningFilesystemSharedCache::getShardForFilename(
    StringRef Filename) const {
  return CacheShards[llvm::hash_value(Filename) % NumShards];
}

DependencyScanningFilesystemSharedCache::CacheShard &
DependencyScanningFilesystemSharedCache::getShardForUID(
    llvm::sys::fs::UniqueID UID) const {
  auto Hash = llvm::hash_combine(UID.getDevice(), UID.getFile());
  return CacheShards[Hash % NumShards];
}

const CachedFileSystemEntry *
DependencyScanningFilesystemSharedCache::CacheShard::findEntryByFilename(
    StringRef Filename) const {
  std::lock_guard<std::mutex> LockGuard(CacheLock);
  auto It = EntriesByFilename.find(Filename);
  return It == EntriesByFilename.end() ? nullptr : It->getValue();
}

const CachedFileSystemEntry *
DependencyScanningFilesystemSharedCache::CacheShard::findEntryByUID(
    llvm::sys::fs::UniqueID UID) const {
  std::lock_guard<std::mutex> LockGuard(CacheLock);
  auto It = EntriesByUID.find(UID);
  return It == EntriesByUID.end() ? nullptr : It->getSecond();
}

const CachedFileSystemEntry &
DependencyScanningFilesystemSharedCache::CacheShard::
    getOrEmplaceEntryForFilename(StringRef Filename,
                                 llvm::ErrorOr<llvm::vfs::Status> Stat) {
  std::lock_guard<std::mutex> LockGuard(CacheLock);
  auto Insertion = EntriesByFilename.insert({Filename, nullptr});
  if (Insertion.second)
    Insertion.first->second =
        new (EntryStorage.Allocate()) CachedFileSystemEntry(std::move(Stat));
  return *Insertion.first->second;
}

const CachedFileSystemEntry &
DependencyScanningFilesystemSharedCache::CacheShard::getOrEmplaceEntryForUID(
    llvm::sys::fs::UniqueID UID, llvm::vfs::Status Stat,
    std::unique_ptr<llvm::MemoryBuffer> Contents) {
  std::lock_guard<std::mutex> LockGuard(CacheLock);
  auto Insertion = EntriesByUID.insert({UID, nullptr});
  if (Insertion.second) {
    CachedFileContents *StoredContents = nullptr;
    if (Contents)
      StoredContents = new (ContentsStorage.Allocate())
          CachedFileContents(std::move(Contents));
    Insertion.first->second = new (EntryStorage.Allocate())
        CachedFileSystemEntry(std::move(Stat), StoredContents);
  }
  return *Insertion.first->second;
}

const CachedFileSystemEntry &
DependencyScanningFilesystemSharedCache::CacheShard::
    getOrInsertEntryForFilename(StringRef Filename,
                                const CachedFileSystemEntry &Entry) {
  std::lock_guard<std::mutex> LockGuard(CacheLock);
  return *EntriesByFilename.insert({Filename, &Entry}).first->getValue();
}

/// Whitelist file extensions that should be minimized, treating no extension as
/// a source file that should be minimized.
///
/// This is kinda hacky, it would be better if we knew what kind of file Clang
/// was expecting instead.
static bool shouldScanForDirectivesBasedOnExtension(StringRef Filename) {
  StringRef Ext = llvm::sys::path::extension(Filename);
  if (Ext.empty())
    return true; // C++ standard library
  return llvm::StringSwitch<bool>(Ext)
      .CasesLower(".c", ".cc", ".cpp", ".c++", ".cxx", true)
      .CasesLower(".h", ".hh", ".hpp", ".h++", ".hxx", true)
      .CasesLower(".m", ".mm", true)
      .CasesLower(".i", ".ii", ".mi", ".mmi", true)
      .CasesLower(".def", ".inc", true)
      .Default(false);
}

static bool shouldCacheStatFailures(StringRef Filename) {
  StringRef Ext = llvm::sys::path::extension(Filename);
  if (Ext.empty())
    return false; // This may be the module cache directory.
  // Only cache stat failures on source files.
  return shouldScanForDirectivesBasedOnExtension(Filename);
}

bool DependencyScanningWorkerFilesystem::shouldScanForDirectives(
    StringRef Filename) {
  return shouldScanForDirectivesBasedOnExtension(Filename);
}

const CachedFileSystemEntry &
DependencyScanningWorkerFilesystem::getOrEmplaceSharedEntryForUID(
    TentativeEntry TEntry) {
  auto &Shard = SharedCache.getShardForUID(TEntry.Status.getUniqueID());
  return Shard.getOrEmplaceEntryForUID(TEntry.Status.getUniqueID(),
                                       std::move(TEntry.Status),
                                       std::move(TEntry.Contents));
}

const CachedFileSystemEntry *
DependencyScanningWorkerFilesystem::findEntryByFilenameWithWriteThrough(
    StringRef Filename) {
  if (const auto *Entry = LocalCache.findEntryByFilename(Filename))
    return Entry;
  auto &Shard = SharedCache.getShardForFilename(Filename);
  if (const auto *Entry = Shard.findEntryByFilename(Filename))
    return &LocalCache.insertEntryForFilename(Filename, *Entry);
  return nullptr;
}

llvm::ErrorOr<const CachedFileSystemEntry &>
DependencyScanningWorkerFilesystem::computeAndStoreResult(StringRef Filename) {
  llvm::ErrorOr<llvm::vfs::Status> Stat = getUnderlyingFS().status(Filename);
  if (!Stat) {
    if (!shouldCacheStatFailures(Filename))
      return Stat.getError();
    const auto &Entry =
        getOrEmplaceSharedEntryForFilename(Filename, Stat.getError());
    return insertLocalEntryForFilename(Filename, Entry);
  }

  if (const auto *Entry = findSharedEntryByUID(*Stat))
    return insertLocalEntryForFilename(Filename, *Entry);

  auto TEntry =
      Stat->isDirectory() ? TentativeEntry(*Stat) : readFile(Filename);

  const CachedFileSystemEntry *SharedEntry = [&]() {
    if (TEntry) {
      const auto &UIDEntry = getOrEmplaceSharedEntryForUID(std::move(*TEntry));
      return &getOrInsertSharedEntryForFilename(Filename, UIDEntry);
    }
    return &getOrEmplaceSharedEntryForFilename(Filename, TEntry.getError());
  }();

  return insertLocalEntryForFilename(Filename, *SharedEntry);
}

llvm::ErrorOr<EntryRef>
DependencyScanningWorkerFilesystem::getOrCreateFileSystemEntry(
    StringRef Filename, bool DisableDirectivesScanning) {
  if (const auto *Entry = findEntryByFilenameWithWriteThrough(Filename))
    return scanForDirectivesIfNecessary(*Entry, Filename,
                                        DisableDirectivesScanning)
        .unwrapError();
  auto MaybeEntry = computeAndStoreResult(Filename);
  if (!MaybeEntry)
    return MaybeEntry.getError();
  return scanForDirectivesIfNecessary(*MaybeEntry, Filename,
                                      DisableDirectivesScanning)
      .unwrapError();
}

llvm::ErrorOr<llvm::vfs::Status>
DependencyScanningWorkerFilesystem::status(const Twine &Path) {
  SmallString<256> OwnedFilename;
  StringRef Filename = Path.toStringRef(OwnedFilename);

  llvm::ErrorOr<EntryRef> Result = getOrCreateFileSystemEntry(Filename);
  if (!Result)
    return Result.getError();
  return Result->getStatus();
}

namespace {

/// The VFS that is used by clang consumes the \c CachedFileSystemEntry using
/// this subclass.
class DepScanFile final : public llvm::vfs::File {
public:
  DepScanFile(std::unique_ptr<llvm::MemoryBuffer> Buffer,
              llvm::vfs::Status Stat)
      : Buffer(std::move(Buffer)), Stat(std::move(Stat)) {}

  static llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> create(EntryRef Entry);

  llvm::ErrorOr<llvm::vfs::Status> status() override { return Stat; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
            bool IsVolatile) override {
    return std::move(Buffer);
  }

  std::error_code close() override { return {}; }

private:
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  llvm::vfs::Status Stat;
};

} // end anonymous namespace

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
DepScanFile::create(EntryRef Entry) {
  assert(!Entry.isError() && "error");

  if (Entry.isDirectory())
    return std::make_error_code(std::errc::is_a_directory);

  auto Result = std::make_unique<DepScanFile>(
      llvm::MemoryBuffer::getMemBuffer(Entry.getContents(),
                                       Entry.getStatus().getName(),
                                       /*RequiresNullTerminator=*/false),
      Entry.getStatus());

  return llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>(
      std::unique_ptr<llvm::vfs::File>(std::move(Result)));
}

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
DependencyScanningWorkerFilesystem::openFileForRead(const Twine &Path) {
  SmallString<256> OwnedFilename;
  StringRef Filename = Path.toStringRef(OwnedFilename);

  llvm::ErrorOr<EntryRef> Result = getOrCreateFileSystemEntry(Filename);
  if (!Result)
    return Result.getError();
  return DepScanFile::create(Result.get());
}
