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
    for (const auto &[Path, State] : Shard.CacheByFilename) {
      const CachedFileSystemEntry *Entry = State.Entry;
      // Skip slots without a resolved entry: real-path-only entries from
      // getRealPath, or uncached negative stats. Runs post-scan, so no
      // in-progress slots remain.
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

namespace {
using InProgressEntry =
    DependencyScanningFilesystemSharedCache::InProgressEntry;
using SlotResolved = llvm::ErrorOr<const CachedFileSystemEntry *>;
using SlotProducer = std::shared_ptr<InProgressEntry>;
using SlotAcquisitionResult = std::variant<SlotResolved, SlotProducer>;

/// Returns a resolved entry if one is already present or in-flight under
/// \p K; otherwise installs a fresh \c InProgressEntry and returns it as a
/// producer slot.
template <typename Map, typename Key>
SlotAcquisitionResult acquireSlot(std::mutex &CacheLock, Map &M, const Key &K) {
  std::shared_ptr<InProgressEntry> Pending;
  {
    std::lock_guard<std::mutex> ShardLock(CacheLock);
    auto &State = M[K];

    // Cache hit.
    if (State.Entry)
      return SlotResolved{State.Entry};

    if (!State.InProgress) {
      State.InProgress = std::make_shared<InProgressEntry>();
      return SlotProducer{State.InProgress};
    }

    // Copy the shared_ptr so the slot survives our wait once the shard lock
    // is released and the producer resets State.InProgress on publish.
    Pending = State.InProgress;
  }

  // Wait off the shard lock so unrelated keys in this shard aren't blocked.
  std::unique_lock<std::mutex> EntryLock(Pending->Mutex);
  Pending->CondVar.wait(EntryLock, [&] { return Pending->Done; });
  return SlotResolved{Pending->Result};
}
} // namespace

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

const CachedFileSystemEntry *
DependencyScanningWorkerFilesystem::resolveUIDThroughSharedCache(
    StringRef OriginalFilename, const llvm::vfs::Status &Stat) {
  auto &UIDShard = Service.getSharedCache().getShardForUID(Stat.getUniqueID());
  auto UIDSlot = acquireSlot(UIDShard.CacheLock, UIDShard.EntriesByUID,
                             Stat.getUniqueID());
  if (auto *Resolved = std::get_if<SlotResolved>(&UIDSlot)) {
    assert(*Resolved && **Resolved &&
           "in-progress UID slot fulfilled without an entry");
    return **Resolved;
  }
  auto UIDProducer = std::move(std::get<SlotProducer>(UIDSlot));

  auto TEntry =
      Stat.isDirectory() ? TentativeEntry(Stat) : readFile(OriginalFilename);

  // Allocate the entry and bind the UID slot under one shard-lock acquisition
  // (BumpPtrAllocator isn't thread-safe). On read-failure, the entry wraps
  // the open error so concurrent UID waiters surface it rather than racing
  // to retry the open.
  const CachedFileSystemEntry *SharedEntry;
  {
    std::lock_guard<std::mutex> ShardLock(UIDShard.CacheLock);
    auto &State = UIDShard.EntriesByUID[Stat.getUniqueID()];
    assert(!State.Entry && "UID slot already published an entry");
    if (TEntry) {
      CachedFileContents *StoredContents = nullptr;
      if (TEntry->Contents)
        StoredContents = new (UIDShard.ContentsStorage.Allocate())
            CachedFileContents(std::move(TEntry->Contents));
      SharedEntry = new (UIDShard.EntryStorage.Allocate())
          CachedFileSystemEntry(std::move(TEntry->Status), StoredContents);
    } else {
      SharedEntry = new (UIDShard.EntryStorage.Allocate())
          CachedFileSystemEntry(TEntry.getError());
    }
    State.Entry = SharedEntry;
    State.InProgress.reset();
  }
  UIDProducer->publish(SharedEntry);
  return SharedEntry;
}

llvm::ErrorOr<const CachedFileSystemEntry *>
DependencyScanningWorkerFilesystem::resolveFilenameThroughSharedCache(
    StringRef OriginalFilename, StringRef FilenameForLookup) {
  assert(llvm::sys::path::is_absolute_gnu(FilenameForLookup));
  auto &FilenameShard =
      Service.getSharedCache().getShardForFilename(FilenameForLookup);
  auto FilenameSlot =
      acquireSlot(FilenameShard.CacheLock, FilenameShard.CacheByFilename,
                  FilenameForLookup);
  if (auto *Resolved = std::get_if<SlotResolved>(&FilenameSlot))
    return *Resolved;
  auto FilenameProducer = std::move(std::get<SlotProducer>(FilenameSlot));

  // Compute the outcome. Three cases:
  //   - Stat succeeded: delegate to the UID resolver.
  //   - Stat failed, cacheable: defer error-entry allocation to the critical
  //     section below so allocate+bind+reset share one shard-lock acquisition.
  //   - Stat failed, not cacheable: publish the error to current waiters but
  //     don't persist; a later separate query re-runs the stat (so a file
  //     created mid-scan becomes visible).
  auto Stat = getUnderlyingFS().status(OriginalFilename);
  const bool ShouldCacheNegativeStat =
      !Stat && Service.getOpts().CacheNegativeStats &&
      shouldCacheNegativeStatsForPath(OriginalFilename);
  llvm::ErrorOr<const CachedFileSystemEntry *> Result = std::error_code{};
  if (Stat)
    Result = resolveUIDThroughSharedCache(OriginalFilename, *Stat);
  else if (!ShouldCacheNegativeStat)
    Result = Stat.getError();

  // Bind the result and reset the in-flight slot under a single critical
  // section. The cached-negative case allocates here so allocate+bind+reset
  // share one shard-lock acquisition.
  {
    std::lock_guard<std::mutex> ShardLock(FilenameShard.CacheLock);
    auto &State = FilenameShard.CacheByFilename[FilenameForLookup];
    assert(!State.Entry && "filename slot already published an entry");
    if (ShouldCacheNegativeStat) {
      auto *Entry = new (FilenameShard.EntryStorage.Allocate())
          CachedFileSystemEntry(Stat.getError());
      State.Entry = Entry;
      Result = Entry;
    } else if (Result) {
      State.Entry = *Result;
    }
    State.InProgress.reset();
  }
  FilenameProducer->publish(Result);
  return Result;
}

llvm::ErrorOr<EntryRef>
DependencyScanningWorkerFilesystem::getOrCreateFileSystemEntry(
    StringRef OriginalFilename) {
  SmallString<256> PathBuf;
  auto FilenameForLookup = tryGetFilenameForLookup(OriginalFilename, PathBuf);
  if (!FilenameForLookup)
    return FilenameForLookup.getError();

  auto &Local = LocalCache[*FilenameForLookup];
  if (Local.File)
    return EntryRef(OriginalFilename, *Local.File).unwrapError();

  auto MaybeEntry =
      resolveFilenameThroughSharedCache(OriginalFilename, *FilenameForLookup);
  if (!MaybeEntry)
    return MaybeEntry.getError();
  Local.File = *MaybeEntry;
  return EntryRef(OriginalFilename, **MaybeEntry).unwrapError();
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
  auto &Local = LocalCache[*FilenameForLookup];
  if (Local.RealPath)
    return HandleCachedRealPath(*Local.RealPath);

  // If we have the result in the shared cache, cache it locally.
  auto &Shard =
      Service.getSharedCache().getShardForFilename(*FilenameForLookup);
  if (const auto *ShardRealPath =
          Shard.findRealPathByFilename(*FilenameForLookup)) {
    Local.RealPath = ShardRealPath;
    return HandleCachedRealPath(*Local.RealPath);
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
  Local.RealPath = &RealPath;
  return HandleCachedRealPath(*Local.RealPath);
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
