//===- DependencyScanningFilesystem.h - clang-scan-deps fs ===---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGFILESYSTEM_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGFILESYSTEM_H

#include "clang/Basic/LLVM.h"
#include "clang/Lex/PreprocessorExcludedConditionalDirectiveSkipMapping.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <mutex>

namespace clang {
namespace tooling {
namespace dependencies {

/// An in-memory representation of a file system entity that is of interest to
/// the dependency scanning filesystem.
///
/// It represents one of the following:
/// - opened file with original contents and a stat value,
/// - opened file with original contents, minimized contents and a stat value,
/// - directory entry with its stat value,
/// - filesystem error,
/// - uninitialized entry with unknown status.
class CachedFileSystemEntry {
public:
  /// Creates an uninitialized entry.
  CachedFileSystemEntry()
      : MaybeStat(llvm::vfs::Status()), MinimizedContentsAccess(nullptr) {}

  /// Initialize the cached file system entry.
  void init(llvm::ErrorOr<llvm::vfs::Status> &&MaybeStatus, StringRef Filename,
            llvm::vfs::FileSystem &FS);

  /// Initialize the entry as file with minimized or original contents.
  ///
  /// The filesystem opens the file even for `stat` calls open to avoid the
  /// issues with stat + open of minimized files that might lead to a
  /// mismatching size of the file.
  llvm::ErrorOr<llvm::vfs::Status> initFile(StringRef Filename,
                                            llvm::vfs::FileSystem &FS);

  /// Minimize contents of the file.
  void minimizeFile();

  /// \returns True if the entry is initialized.
  bool isInitialized() const {
    return !MaybeStat || MaybeStat->isStatusKnown();
  }

  /// \returns True if the entry is a filesystem error.
  bool isError() const { return !MaybeStat; }

  /// \returns True if the current entry points to a directory.
  bool isDirectory() const { return !isError() && MaybeStat->isDirectory(); }

  /// \returns Original contents of the file.
  StringRef getOriginalContents() const {
    assert(isInitialized() && "not initialized");
    assert(!isError() && "error");
    assert(!MaybeStat->isDirectory() && "not a file");
    assert(OriginalContents && "not read");
    return OriginalContents->getBuffer();
  }

  /// \returns Minimized contents of the file.
  StringRef getMinimizedContents() const {
    assert(isInitialized() && "not initialized");
    assert(!isError() && "error");
    assert(!isDirectory() && "not a file");
    llvm::MemoryBuffer *Buffer = MinimizedContentsAccess.load();
    assert(Buffer && "not minimized");
    return Buffer->getBuffer();
  }

  /// \returns True if this entry represents a file that can be read.
  bool isReadable() const { return MaybeStat && !MaybeStat->isDirectory(); }

  /// \returns True if this cached entry needs to be updated.
  bool needsUpdate(bool ShouldBeMinimized) const {
    return isReadable() && needsMinimization(ShouldBeMinimized);
  }

  /// \returns True if the contents of this entry need to be minimized.
  bool needsMinimization(bool ShouldBeMinimized) const {
    return ShouldBeMinimized && !MinimizedContentsAccess.load();
  }

  /// \returns The error.
  std::error_code getError() const {
    assert(isInitialized() && "not initialized");
    return MaybeStat.getError();
  }

  /// \returns The entry status.
  llvm::vfs::Status getStatus() const {
    assert(isInitialized() && "not initialized");
    assert(!isError() && "error");
    return *MaybeStat;
  }

  /// \returns the name of the file.
  StringRef getName() const {
    assert(isInitialized() && "not initialized");
    assert(!isError() && "error");
    return MaybeStat->getName();
  }

  /// Return the mapping between location -> distance that is used to speed up
  /// the block skipping in the preprocessor.
  const PreprocessorSkippedRangeMapping &getPPSkippedRangeMapping() const {
    assert(!isError() && "error");
    assert(!isDirectory() && "not a file");
    return PPSkippedRangeMapping;
  }

private:
  llvm::ErrorOr<llvm::vfs::Status> MaybeStat;
  std::unique_ptr<llvm::MemoryBuffer> OriginalContents;

  /// Owning storage for the minimized file contents.
  std::unique_ptr<llvm::MemoryBuffer> MinimizedContentsStorage;
  /// Atomic view of the minimized file contents.
  /// This prevents data races when multiple threads call `needsMinimization`.
  std::atomic<llvm::MemoryBuffer *> MinimizedContentsAccess;

  PreprocessorSkippedRangeMapping PPSkippedRangeMapping;
};

/// This class is a shared cache, that caches the 'stat' and 'open' calls to the
/// underlying real file system. It distinguishes between minimized and original
/// files.
///
/// It is sharded based on the hash of the key to reduce the lock contention for
/// the worker threads.
class DependencyScanningFilesystemSharedCache {
public:
  struct SharedFileSystemEntry {
    std::mutex ValueLock;
    CachedFileSystemEntry Value;
  };

  DependencyScanningFilesystemSharedCache();

  /// Returns a cache entry for the corresponding key.
  ///
  /// A new cache entry is created if the key is not in the cache. This is a
  /// thread safe call.
  SharedFileSystemEntry &get(StringRef Key);

private:
  struct CacheShard {
    std::mutex CacheLock;
    llvm::StringMap<SharedFileSystemEntry, llvm::BumpPtrAllocator> Cache;
  };
  std::unique_ptr<CacheShard[]> CacheShards;
  unsigned NumShards;
};

/// This class is a local cache, that caches the 'stat' and 'open' calls to the
/// underlying real file system. It distinguishes between minimized and original
/// files.
class DependencyScanningFilesystemLocalCache {
  llvm::StringMap<const CachedFileSystemEntry *, llvm::BumpPtrAllocator> Cache;

public:
  const CachedFileSystemEntry *getCachedEntry(StringRef Filename) {
    return Cache[Filename];
  }
};

/// Reference to a CachedFileSystemEntry.
/// If the underlying entry is an opened file, this wrapper returns the correct
/// contents (original or minimized) and ensures consistency with file size
/// reported by status.
class EntryRef {
  /// For entry that is an opened file, this bit signifies whether its contents
  /// are minimized.
  bool Minimized;

  /// The underlying cached entry.
  const CachedFileSystemEntry &Entry;

public:
  EntryRef(bool Minimized, const CachedFileSystemEntry &Entry)
      : Minimized(Minimized), Entry(Entry) {}

  llvm::vfs::Status getStatus() const {
    llvm::vfs::Status Stat = Entry.getStatus();
    if (Stat.isDirectory())
      return Stat;
    return llvm::vfs::Status::copyWithNewSize(Stat, getContents().size());
  }

  bool isError() const { return Entry.isError(); }
  bool isDirectory() const { return Entry.isDirectory(); }
  StringRef getName() const { return Entry.getName(); }

  /// If the cached entry represents an error, promotes it into `ErrorOr`.
  llvm::ErrorOr<EntryRef> unwrapError() const {
    if (isError())
      return Entry.getError();
    return *this;
  }

  StringRef getContents() const {
    return Minimized ? Entry.getMinimizedContents()
                     : Entry.getOriginalContents();
  }

  const PreprocessorSkippedRangeMapping *getPPSkippedRangeMapping() const {
    return Minimized ? &Entry.getPPSkippedRangeMapping() : nullptr;
  }
};

/// A virtual file system optimized for the dependency discovery.
///
/// It is primarily designed to work with source files whose contents was was
/// preprocessed to remove any tokens that are unlikely to affect the dependency
/// computation.
///
/// This is not a thread safe VFS. A single instance is meant to be used only in
/// one thread. Multiple instances are allowed to service multiple threads
/// running in parallel.
class DependencyScanningWorkerFilesystem : public llvm::vfs::ProxyFileSystem {
public:
  DependencyScanningWorkerFilesystem(
      DependencyScanningFilesystemSharedCache &SharedCache,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
      ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings)
      : ProxyFileSystem(std::move(FS)), SharedCache(SharedCache),
        PPSkipMappings(PPSkipMappings) {}

  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override;
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const Twine &Path) override;

  /// Disable minimization of the given file.
  void disableMinimization(StringRef Filename);
  /// Enable minimization of all files.
  void enableMinimizationOfAllFiles() { NotToBeMinimized.clear(); }

private:
  /// Check whether the file should be minimized.
  bool shouldMinimize(StringRef Filename);

  llvm::ErrorOr<EntryRef> getOrCreateFileSystemEntry(StringRef Filename);

  /// The global cache shared between worker threads.
  DependencyScanningFilesystemSharedCache &SharedCache;
  /// The local cache is used by the worker thread to cache file system queries
  /// locally instead of querying the global cache every time.
  DependencyScanningFilesystemLocalCache LocalCache;
  /// The optional mapping structure which records information about the
  /// excluded conditional directive skip mappings that are used by the
  /// currently active preprocessor.
  ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings;
  /// The set of files that should not be minimized.
  llvm::StringSet<> NotToBeMinimized;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGFILESYSTEM_H
