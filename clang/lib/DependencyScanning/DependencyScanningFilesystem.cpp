//===- DependencyScanningFilesystem.cpp - Optimized Scanning FS -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"
#include <optional>

using namespace clang;
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

bool DependencyScanningWorkerFilesystem::ensureDirectiveTokensArePopulated(
    EntryRef Ref) {
  auto &Entry = Ref.Entry;

  if (Entry.isError() || Entry.isDirectory())
    return false;

  CachedFileContents *Contents = Entry.getCachedContents();
  assert(Contents && "contents not initialized");

  // Double-checked locking.
  if (Contents->DepDirectives.load())
    return true;

  std::lock_guard<std::mutex> GuardLock(Contents->ValueLock);

  // Double-checked locking.
  if (Contents->DepDirectives.load())
    return true;

  SmallVector<dependency_directives_scan::Directive, 64> Directives;
  // Scan the file for preprocessor directives that might affect the
  // dependencies.
  if (scanSourceForDependencyDirectives(Contents->Original->getBuffer(),
                                        Contents->DepDirectiveTokens,
                                        Directives)) {
    Contents->DepDirectiveTokens.clear();
    // FIXME: Propagate the diagnostic if desired by the client.
    Contents->DepDirectives.store(new std::optional<DependencyDirectivesTy>());
    return false;
  }

  // This function performed double-checked locking using `DepDirectives`.
  // Assigning it must be the last thing this function does, otherwise other
  // threads may skip the critical section (`DepDirectives != nullptr`), leading
  // to a data race.
  Contents->DepDirectives.store(
      new std::optional<DependencyDirectivesTy>(std::move(Directives)));
  return true;
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
  assert(llvm::sys::path::is_absolute_gnu(Filename));
  return CacheShards[llvm::hash_value(Filename) % NumShards];
}

DependencyScanningFilesystemSharedCache::CacheShard &
DependencyScanningFilesystemSharedCache::getShardForUID(
    llvm::sys::fs::UniqueID UID) const {
  auto Hash = llvm::hash_combine(UID.getDevice(), UID.getFile());
  return CacheShards[Hash % NumShards];
}

std::vector<DependencyScanningFilesystemSharedCache::OutOfDateEntry>
DependencyScanningFilesystemSharedCache::getOutOfDateEntries(
    llvm::vfs::FileSystem &UnderlyingFS) const {
  // Iterate through all shards and look for cached stat errors.
  std::vector<OutOfDateEntry> InvalidDiagInfo;
  for (unsigned i = 0; i < NumShards; i++) {
    const CacheShard &Shard = CacheShards[i];
    std::lock_guard<std::mutex> LockGuard(Shard.CacheLock);
    for (const auto &[Path, CachedPair] : Shard.CacheByFilename) {
      const auto &Entry = CachedPair.File;
      // Stat failure that wasn't cached.
      if (!Entry)
        continue;
      llvm::ErrorOr<llvm::vfs::Status> Status = UnderlyingFS.status(Path);
      if (Status) {
        if (Entry->getError()) {
          // This is the case where we have cached the non-existence
          // of the file at Path first, and a file at the path is created
          // later. The cache entry is not invalidated (as we have no good
          // way to do it now), which may lead to missing file build errors.
          InvalidDiagInfo.emplace_back(Path.data());
        } else {
          llvm::vfs::Status CachedStatus = Entry->getStatus();
          if (Status->getType() == llvm::sys::fs::file_type::regular_file &&
              Status->getType() == CachedStatus.getType()) {
            // We only check regular files. Directory files sizes could change
            // due to content changes, and reporting directory size changes can
            // lead to false positives.
            // TODO: At the moment, we do not detect symlinks to files whose
            // size may change. We need to decide if we want to detect cached
            // symlink size changes. We can also expand this to detect file
            // type changes.
            uint64_t CachedSize = CachedStatus.getSize();
            uint64_t ActualSize = Status->getSize();
            if (CachedSize != ActualSize) {
              // This is the case where the cached file has a different size
              // from the actual file that comes from the underlying FS.
              InvalidDiagInfo.emplace_back(Path.data(), CachedSize, ActualSize);
            }
          }
        }
      }
    }
  }
  return InvalidDiagInfo;
}

const CachedRealPath *
DependencyScanningFilesystemSharedCache::CacheShard::findRealPathByFilename(
    StringRef Filename) const {
  assert(llvm::sys::path::is_absolute_gnu(Filename));
  std::lock_guard<std::mutex> LockGuard(CacheLock);
  auto It = CacheByFilename.find(Filename);
  return It == CacheByFilename.end() ? nullptr : It->getValue().RealPath;
}

const CachedRealPath &DependencyScanningFilesystemSharedCache::CacheShard::
    getOrEmplaceRealPathForFilename(StringRef Filename,
                                    llvm::ErrorOr<llvm::StringRef> RealPath) {
  std::lock_guard<std::mutex> LockGuard(CacheLock);

  const CachedRealPath *&StoredRealPath = CacheByFilename[Filename].RealPath;
  if (!StoredRealPath) {
    auto OwnedRealPath = [&]() -> CachedRealPath {
      if (!RealPath)
        return RealPath.getError();
      return RealPath->str();
    }();

    StoredRealPath = new (RealPathStorage.Allocate())
        CachedRealPath(std::move(OwnedRealPath));
  }

  return *StoredRealPath;
}

DependencyScanningWorkerFilesystem::DependencyScanningWorkerFilesystem(
    DependencyScanningService &Service,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : llvm::RTTIExtends<DependencyScanningWorkerFilesystem,
                        llvm::vfs::ProxyFileSystem>(std::move(FS)),
      Service(Service), WorkingDirForCacheLookup(llvm::errc::invalid_argument) {
  updateWorkingDirForCacheLookup();
}

llvm::ErrorOr<EntryRef>
DependencyScanningWorkerFilesystem::getOrCreateFileSystemEntry(
    StringRef OriginalFilename) {
  SmallString<256> PathBuf;
  auto FilenameForLookup = tryGetFilenameForLookup(OriginalFilename, PathBuf);
  if (!FilenameForLookup)
    return FilenameForLookup.getError();

  auto &LocalEntry = LocalCache[*FilenameForLookup];
  if (LocalEntry.File)
    return EntryRef(OriginalFilename, *LocalEntry.File).unwrapError();

  auto &Shard =
      Service.getSharedCache().getShardForFilename(*FilenameForLookup);

  auto &SharedEntry = [&]()
      -> DependencyScanningFilesystemSharedCache::CacheShard::SharedEntry & {
    std::lock_guard<std::mutex> LockGuard(Shard.CacheLock);
    auto [It, Inserted] = Shard.CacheByFilename.try_emplace(*FilenameForLookup);
    auto &[Key, Value] = *It;
    return Value;
  }();

  // Operations on SharedEntry.Value must be atomic, because here we're trying
  // to avoid contending the mutex by checking outside the critical section.
  if (auto SharedEntryValue = std::atomic_load(&SharedEntry.File)) {
    LocalEntry.File = std::move(SharedEntryValue);
    return EntryRef(OriginalFilename, *LocalEntry.File).unwrapError();
  }

  std::lock_guard<std::mutex> LockGuard(SharedEntry.Mutex);
  if (auto SharedEntryValue = std::atomic_load(&SharedEntry.File)) {
    LocalEntry.File = std::move(SharedEntryValue);
    return EntryRef(OriginalFilename, *LocalEntry.File).unwrapError();
  }

  llvm::ErrorOr<llvm::vfs::Status> Stat =
    getUnderlyingFS().status(OriginalFilename);

  if (!Stat) {
    // Note that we're leaving behind local and shared entry with nullptr File.
    if (!Service.getOpts().CacheNegativeStats ||
        !shouldCacheNegativeStatsForPath(OriginalFilename))
      return Stat.getError();

    std::shared_ptr<CachedFileSystemEntry> Expected = nullptr;
    std::atomic_compare_exchange_strong(
        &SharedEntry.File, &Expected,
        std::make_shared<CachedFileSystemEntry>(std::move(Stat)));
    assert(Expected == nullptr && "Concurrent FS access for identical path");
    LocalEntry.File = std::atomic_load(&SharedEntry.File);
    return EntryRef(OriginalFilename, *LocalEntry.File).unwrapError();
  }

  auto &SharedUIDEntry = [&]() -> std::shared_ptr<CachedFileSystemEntry> & {
    std::lock_guard<std::mutex> LockGuard(Shard.CacheLock);
    auto &Wrapper = Shard.EntriesByUID[Stat->getUniqueID()];
    if (!Wrapper)
      Wrapper = std::make_unique<std::shared_ptr<CachedFileSystemEntry>>();
    return *Wrapper;
  }();

  // Operations on SharedUIDEntry must be atomic, because the lock above only
  // guarantees mutual exclusion for identical paths. If we have concurrent
  // access to the same file using different file names, we're still doing
  // this concurrently. This is rare enough that ensuring mutual exclusion is
  // unlikely beneficial.
  if (!std::atomic_load(&SharedUIDEntry)) {
    auto TEntry =
      Stat->isDirectory() ? TentativeEntry(*Stat) : readFile(OriginalFilename);

    std::shared_ptr<CachedFileSystemEntry> NewEntry = [&]() {
      if (!TEntry)
        return std::make_shared<CachedFileSystemEntry>(TEntry.getError());

      std::unique_ptr<CachedFileContents> StoredContents =
          TEntry->Contents ? std::make_unique<CachedFileContents>(
                                 std::move(TEntry->Contents))
                           : nullptr;
      return std::make_shared<CachedFileSystemEntry>(
          std::move(TEntry->Status), std::move(StoredContents));
    }();

    std::shared_ptr<CachedFileSystemEntry> Expected = nullptr;
    std::atomic_compare_exchange_strong(&SharedUIDEntry, &Expected,
                                        std::move(NewEntry));
    // If Expected != nullptr, we had a race between two different file names
    // resolving to the same file.
  }

  // Propagate the entry from the UID map.
  std::shared_ptr<CachedFileSystemEntry> Expected = nullptr;
  std::atomic_compare_exchange_strong(&SharedEntry.File, &Expected,
                                      std::atomic_load(&SharedUIDEntry));
  assert(Expected == nullptr && "Concurrent FS access for identical path");

  LocalEntry.File = std::atomic_load(&SharedEntry.File);

  return EntryRef(OriginalFilename, *LocalEntry.File).unwrapError();
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

bool DependencyScanningWorkerFilesystem::exists(const Twine &Path) {
  // While some VFS overlay filesystems may implement more-efficient
  // mechanisms for `exists` queries, `DependencyScanningWorkerFilesystem`
  // typically wraps `RealFileSystem` which does not specialize `exists`,
  // so it is not likely to benefit from such optimizations. Instead,
  // it is more-valuable to have this query go through the
  // cached-`status` code-path of the `DependencyScanningWorkerFilesystem`.
  llvm::ErrorOr<llvm::vfs::Status> Status = status(Path);
  return Status && Status->exists();
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
    return llvm::MemoryBuffer::getMemBuffer(Buffer->getMemBufferRef(),
                                            RequiresNullTerminator);
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

std::error_code
DependencyScanningWorkerFilesystem::getRealPath(const Twine &Path,
                                                SmallVectorImpl<char> &Output) {
  SmallString<256> OwnedFilename;
  StringRef OriginalFilename = Path.toStringRef(OwnedFilename);

  SmallString<256> PathBuf;
  auto FilenameForLookup = tryGetFilenameForLookup(OriginalFilename, PathBuf);
  if (!FilenameForLookup)
    return FilenameForLookup.getError();

  auto HandleCachedRealPath =
      [&Output](const CachedRealPath &RealPath) -> std::error_code {
    if (!RealPath)
      return RealPath.getError();
    Output.assign(RealPath->begin(), RealPath->end());
    return {};
  };

  // If we already have the result in local cache, no work required.
  auto &LocalEntry = LocalCache[*FilenameForLookup];
  if (LocalEntry.RealPath)
    return HandleCachedRealPath(*LocalEntry.RealPath);

  // If we have the result in the shared cache, cache it locally.
  auto &Shard =
      Service.getSharedCache().getShardForFilename(*FilenameForLookup);
  if (const auto *ShardRealPath =
          Shard.findRealPathByFilename(*FilenameForLookup)) {
    LocalEntry.RealPath = ShardRealPath;
    return HandleCachedRealPath(*LocalEntry.RealPath);
  }

  // If we don't know the real path, compute it...
  std::error_code EC = getUnderlyingFS().getRealPath(OriginalFilename, Output);
  llvm::ErrorOr<llvm::StringRef> ComputedRealPath = EC;
  if (!EC)
    ComputedRealPath = StringRef{Output.data(), Output.size()};

  // ...and try to write it into the shared cache. In case some other thread won
  // this race and already wrote its own result there, just adopt it. Write
  // whatever is in the shared cache into the local one.
  const auto &RealPath = Shard.getOrEmplaceRealPathForFilename(
      *FilenameForLookup, ComputedRealPath);
  LocalEntry.RealPath = &RealPath;
  return HandleCachedRealPath(*LocalEntry.RealPath);
}

std::error_code DependencyScanningWorkerFilesystem::setCurrentWorkingDirectory(
    const Twine &Path) {
  std::error_code EC = ProxyFileSystem::setCurrentWorkingDirectory(Path);
  updateWorkingDirForCacheLookup();
  return EC;
}

void DependencyScanningWorkerFilesystem::updateWorkingDirForCacheLookup() {
  llvm::ErrorOr<std::string> CWD =
      getUnderlyingFS().getCurrentWorkingDirectory();
  if (!CWD) {
    WorkingDirForCacheLookup = CWD.getError();
  } else if (!llvm::sys::path::is_absolute_gnu(*CWD)) {
    WorkingDirForCacheLookup = llvm::errc::invalid_argument;
  } else {
    WorkingDirForCacheLookup = *CWD;
  }
  assert(!WorkingDirForCacheLookup ||
         llvm::sys::path::is_absolute_gnu(*WorkingDirForCacheLookup));
}

llvm::ErrorOr<StringRef>
DependencyScanningWorkerFilesystem::tryGetFilenameForLookup(
    StringRef OriginalFilename, llvm::SmallVectorImpl<char> &PathBuf) const {
  StringRef FilenameForLookup;
  if (llvm::sys::path::is_absolute_gnu(OriginalFilename)) {
    FilenameForLookup = OriginalFilename;
  } else if (!WorkingDirForCacheLookup) {
    return WorkingDirForCacheLookup.getError();
  } else {
    StringRef RelFilename = OriginalFilename;
    RelFilename.consume_front("./");
    PathBuf.assign(WorkingDirForCacheLookup->begin(),
                   WorkingDirForCacheLookup->end());
    llvm::sys::path::append(PathBuf, RelFilename);
    FilenameForLookup = StringRef{PathBuf.begin(), PathBuf.size()};
  }
  assert(llvm::sys::path::is_absolute_gnu(FilenameForLookup));
  return FilenameForLookup;
}

const char DependencyScanningWorkerFilesystem::ID = 0;
