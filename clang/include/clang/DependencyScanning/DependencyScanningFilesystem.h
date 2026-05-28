//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGFILESYSTEM_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGFILESYSTEM_H

#include "clang/Basic/LLVM.h"
#include "clang/Lex/DependencyDirectivesScanner.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <mutex>
#include <optional>
#include <variant>

namespace clang {
namespace dependencies {

class DependencyScanningService;

using DependencyDirectivesTy =
    SmallVector<dependency_directives_scan::Directive, 20>;

/// Contents and directive tokens of a cached file entry. Single instance can
/// be shared between multiple entries.
struct CachedFileContents {
  CachedFileContents(std::unique_ptr<llvm::MemoryBuffer> Contents)
      : Original(std::move(Contents)), DepDirectives(nullptr) {}

  /// Owning storage for the original contents.
  std::unique_ptr<llvm::MemoryBuffer> Original;

  /// The mutex that must be locked before mutating directive tokens.
  std::mutex ValueLock;
  SmallVector<dependency_directives_scan::Token, 10> DepDirectiveTokens;
  /// Accessor to the directive tokens that's atomic to avoid data races.
  /// \p CachedFileContents has ownership of the pointer.
  std::atomic<const std::optional<DependencyDirectivesTy> *> DepDirectives;

  ~CachedFileContents() { delete DepDirectives.load(); }
};

/// An in-memory representation of a file system entity that is of interest to
/// the dependency scanning filesystem.
///
/// It represents one of the following:
/// - opened file with contents and a stat value,
/// - opened file with contents, directive tokens and a stat value,
/// - directory entry with its stat value,
/// - filesystem error.
///
/// Single instance of this class can be shared across different filenames (e.g.
/// a regular file and a symlink). For this reason the status filename is empty
/// and is only materialized by \c EntryRef that knows the requested filename.
class CachedFileSystemEntry {
public:
  /// Creates an entry without contents: either a filesystem error or
  /// a directory with stat value.
  CachedFileSystemEntry(llvm::ErrorOr<llvm::vfs::Status> Stat)
      : MaybeStat(std::move(Stat)), Contents(nullptr) {
    clearStatName();
  }

  /// Creates an entry representing a file with contents.
  CachedFileSystemEntry(llvm::ErrorOr<llvm::vfs::Status> Stat,
                        std::unique_ptr<CachedFileContents> Contents)
      : MaybeStat(std::move(Stat)), Contents(std::move(Contents)) {
    clearStatName();
  }

  /// \returns True if the entry is a filesystem error.
  bool isError() const { return !MaybeStat; }

  /// \returns True if the current entry represents a directory.
  bool isDirectory() const { return !isError() && MaybeStat->isDirectory(); }

  /// \returns Original contents of the file.
  StringRef getOriginalContents() const {
    assert(!isError() && "error");
    assert(!MaybeStat->isDirectory() && "not a file");
    assert(Contents && "contents not initialized");
    return Contents->Original->getBuffer();
  }

  /// \returns The scanned preprocessor directive tokens of the file that are
  /// used to speed up preprocessing, if available.
  std::optional<ArrayRef<dependency_directives_scan::Directive>>
  getDirectiveTokens() const {
    assert(!isError() && "error");
    assert(!isDirectory() && "not a file");
    assert(Contents && "contents not initialized");
    if (auto *Directives = Contents->DepDirectives.load()) {
      if (Directives->has_value())
        return ArrayRef<dependency_directives_scan::Directive>(**Directives);
    }
    return std::nullopt;
  }

  /// \returns The error.
  std::error_code getError() const { return MaybeStat.getError(); }

  /// \returns The entry status with empty filename.
  llvm::vfs::Status getStatus() const {
    assert(!isError() && "error");
    assert(MaybeStat->getName().empty() && "stat name must be empty");
    return *MaybeStat;
  }

  /// \returns The unique ID of the entry.
  llvm::sys::fs::UniqueID getUniqueID() const {
    assert(!isError() && "error");
    return MaybeStat->getUniqueID();
  }

  /// \returns The data structure holding both contents and directive tokens.
  CachedFileContents *getCachedContents() const {
    assert(!isError() && "error");
    assert(!isDirectory() && "not a file");
    return Contents.get();
  }

private:
  void clearStatName() {
    if (MaybeStat)
      MaybeStat = llvm::vfs::Status::copyWithNewName(*MaybeStat, "");
  }

  /// Either the filesystem error or status of the entry.
  /// The filename is empty and only materialized by \c EntryRef.
  llvm::ErrorOr<llvm::vfs::Status> MaybeStat;

  /// We're using pointer here to keep the size of this class small. Instances
  /// representing directories and filesystem errors don't hold any contents
  /// anyway.
  std::unique_ptr<CachedFileContents> Contents;
};

using CachedRealPath = llvm::ErrorOr<std::string>;

/// This class is a shared cache, that caches the 'stat' and 'open' calls to the
/// underlying real file system, and the scanned preprocessor directives of
/// files.
///
/// It is sharded based on the hash of the key to reduce the lock contention for
/// the worker threads.
class DependencyScanningFilesystemSharedCache {
public:
  struct CacheShard {
    /// The mutex that needs to be locked before mutation of any member.
    mutable std::mutex CacheLock;

    struct SharedEntry {
      std::mutex Mutex;
      std::shared_ptr<CachedFileSystemEntry> File = nullptr;

      const CachedRealPath *RealPath = nullptr;
    };

    /// Map from filenames to cached entries and real paths.
    llvm::StringMap<SharedEntry, llvm::BumpPtrAllocator> CacheByFilename;

    /// Map from unique IDs to cached entries.
    llvm::DenseMap<llvm::sys::fs::UniqueID,
                   std::unique_ptr<std::shared_ptr<CachedFileSystemEntry>>>
        EntriesByUID;

    /// The backing storage for cached real paths.
    llvm::SpecificBumpPtrAllocator<CachedRealPath> RealPathStorage;

    /// Returns the real path associated with the filename or nullptr if none is
    /// found.
    const CachedRealPath *findRealPathByFilename(StringRef Filename) const;

    /// Returns the real path associated with the filename if there is some.
    /// Otherwise, constructs new one with the given one, associates it with the
    /// filename and returns the result.
    const CachedRealPath &
    getOrEmplaceRealPathForFilename(StringRef Filename,
                                    llvm::ErrorOr<StringRef> RealPath);
  };

  DependencyScanningFilesystemSharedCache();

  /// Returns shard for the given key.
  CacheShard &getShardForFilename(StringRef Filename) const;
  CacheShard &getShardForUID(llvm::sys::fs::UniqueID UID) const;

  struct OutOfDateEntry {
    // A null terminated string that contains a path.
    const char *Path = nullptr;

    struct NegativelyCachedInfo {};
    struct SizeChangedInfo {
      uint64_t CachedSize = 0;
      uint64_t ActualSize = 0;
    };

    std::variant<NegativelyCachedInfo, SizeChangedInfo> Info;

    OutOfDateEntry(const char *Path)
        : Path(Path), Info(NegativelyCachedInfo{}) {}

    OutOfDateEntry(const char *Path, uint64_t CachedSize, uint64_t ActualSize)
        : Path(Path), Info(SizeChangedInfo{CachedSize, ActualSize}) {}
  };

  /// Visits all cached entries and re-stat an entry using UnderlyingFS to check
  /// if the cache contains out-of-date entries. An entry can be out-of-date for
  /// two reasons:
  ///  1. The entry contains a stat error, indicating the file did not exist
  ///     in the cache, but the file exists on the UnderlyingFS.
  ///  2. The entry is associated with a file whose size is different from the
  ///     size of the file on the same path on the UnderlyingFS.
  std::vector<OutOfDateEntry>
  getOutOfDateEntries(llvm::vfs::FileSystem &UnderlyingFS) const;

private:
  std::unique_ptr<CacheShard[]> CacheShards;
  unsigned NumShards;
};

/// Reference to a CachedFileSystemEntry.
/// If the underlying entry is an opened file, this wrapper returns the file
/// contents and the scanned preprocessor directives.
class EntryRef {
  /// The filename used to access this entry.
  std::string Filename;

  /// The underlying cached entry.
  const CachedFileSystemEntry &Entry;

  friend class DependencyScanningWorkerFilesystem;

public:
  EntryRef(StringRef Name, const CachedFileSystemEntry &Entry)
      : Filename(Name), Entry(Entry) {}

  llvm::vfs::Status getStatus() const {
    llvm::vfs::Status Stat = Entry.getStatus();
    if (!Stat.isDirectory())
      Stat = llvm::vfs::Status::copyWithNewSize(Stat, getContents().size());
    return llvm::vfs::Status::copyWithNewName(Stat, Filename);
  }

  bool isError() const { return Entry.isError(); }
  bool isDirectory() const { return Entry.isDirectory(); }

  /// If the cached entry represents an error, promotes it into `ErrorOr`.
  llvm::ErrorOr<EntryRef> unwrapError() const {
    if (isError())
      return Entry.getError();
    return *this;
  }

  StringRef getContents() const { return Entry.getOriginalContents(); }

  std::optional<ArrayRef<dependency_directives_scan::Directive>>
  getDirectiveTokens() const {
    return Entry.getDirectiveTokens();
  }
};

/// A virtual file system optimized for the dependency discovery.
///
/// It is primarily designed to work with source files whose contents was
/// preprocessed to remove any tokens that are unlikely to affect the dependency
/// computation.
///
/// This is not a thread safe VFS. A single instance is meant to be used only in
/// one thread. Multiple instances are allowed to service multiple threads
/// running in parallel.
class DependencyScanningWorkerFilesystem
    : public llvm::RTTIExtends<DependencyScanningWorkerFilesystem,
                               llvm::vfs::ProxyFileSystem> {
public:
  static const char ID;

  DependencyScanningWorkerFilesystem(
      DependencyScanningService &Service,
      IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override;
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const Twine &Path) override;

  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) override;

  std::error_code setCurrentWorkingDirectory(const Twine &Path) override;

  /// Returns entry for the given filename.
  ///
  /// Attempts to use the local and shared caches first, then falls back to
  /// using the underlying filesystem.
  llvm::ErrorOr<EntryRef> getOrCreateFileSystemEntry(StringRef Filename);

  /// Ensure the directive tokens are populated for this file entry.
  ///
  /// Returns true if the directive tokens are populated for this file entry,
  /// false if not (i.e. this entry is not a file or its scan fails).
  bool ensureDirectiveTokensArePopulated(EntryRef Entry);

  /// \returns The scanned preprocessor directive tokens of the file that are
  /// used to speed up preprocessing, if available.
  std::optional<ArrayRef<dependency_directives_scan::Directive>>
  getDirectiveTokens(const Twine &Path) {
    if (llvm::ErrorOr<EntryRef> Entry = getOrCreateFileSystemEntry(Path.str()))
      if (ensureDirectiveTokensArePopulated(*Entry))
        return Entry->getDirectiveTokens();
    return std::nullopt;
  }

  /// Check whether \p Path exists. By default checks cached result of \c
  /// status(), and falls back on FS if unable to do so.
  bool exists(const Twine &Path) override;

private:
  /// Represents a filesystem entry that has been stat-ed (and potentially read)
  /// and that's about to be inserted into the cache as `CachedFileSystemEntry`.
  struct TentativeEntry {
    llvm::vfs::Status Status;
    std::unique_ptr<llvm::MemoryBuffer> Contents;

    TentativeEntry(llvm::vfs::Status Status,
                   std::unique_ptr<llvm::MemoryBuffer> Contents = nullptr)
        : Status(std::move(Status)), Contents(std::move(Contents)) {}
  };

  /// Reads file at the given path. Enforces consistency between the file size
  /// in status and size of read contents.
  llvm::ErrorOr<TentativeEntry> readFile(StringRef Filename);

  void printImpl(raw_ostream &OS, PrintType Type,
                 unsigned IndentLevel) const override {
    printIndent(OS, IndentLevel);
    OS << "DependencyScanningFilesystem\n";
    getUnderlyingFS().print(OS, Type, IndentLevel + 1);
  }

  /// The service associated with this VFS.
  DependencyScanningService &Service;

  struct LocalEntry {
    std::shared_ptr<CachedFileSystemEntry> File = nullptr;
    const CachedRealPath *RealPath = nullptr;
  };

  /// The local cache is used by the worker thread to cache file system queries
  /// locally instead of querying the global cache every time.
  llvm::StringMap<LocalEntry, llvm::BumpPtrAllocator> LocalCache;

  /// The working directory to use for making relative paths absolute before
  /// using them for cache lookups.
  llvm::ErrorOr<std::string> WorkingDirForCacheLookup;

  void updateWorkingDirForCacheLookup();

  llvm::ErrorOr<StringRef>
  tryGetFilenameForLookup(StringRef OriginalFilename,
                          llvm::SmallVectorImpl<char> &PathBuf) const;
};

} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGFILESYSTEM_H
