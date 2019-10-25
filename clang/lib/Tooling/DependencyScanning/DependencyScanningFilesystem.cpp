//===- DependencyScanningFilesystem.cpp - clang-scan-deps fs --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/Lex/DependencyDirectivesSourceMinimizer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

CachedFileSystemEntry CachedFileSystemEntry::createFileEntry(
    StringRef Filename, llvm::vfs::FileSystem &FS, bool Minimize) {
  // Load the file and its content from the file system.
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> MaybeFile =
      FS.openFileForRead(Filename);
  if (!MaybeFile)
    return MaybeFile.getError();
  llvm::ErrorOr<llvm::vfs::Status> Stat = (*MaybeFile)->status();
  if (!Stat)
    return Stat.getError();

  llvm::vfs::File &F = **MaybeFile;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MaybeBuffer =
      F.getBuffer(Stat->getName());
  if (!MaybeBuffer)
    return MaybeBuffer.getError();

  llvm::SmallString<1024> MinimizedFileContents;
  // Minimize the file down to directives that might affect the dependencies.
  const auto &Buffer = *MaybeBuffer;
  SmallVector<minimize_source_to_dependency_directives::Token, 64> Tokens;
  if (!Minimize || minimizeSourceToDependencyDirectives(
                       Buffer->getBuffer(), MinimizedFileContents, Tokens)) {
    // Use the original file unless requested otherwise, or
    // if the minimization failed.
    // FIXME: Propage the diagnostic if desired by the client.
    CachedFileSystemEntry Result;
    Result.MaybeStat = std::move(*Stat);
    Result.Contents.reserve(Buffer->getBufferSize() + 1);
    Result.Contents.append(Buffer->getBufferStart(), Buffer->getBufferEnd());
    // Implicitly null terminate the contents for Clang's lexer.
    Result.Contents.push_back('\0');
    Result.Contents.pop_back();
    return Result;
  }

  CachedFileSystemEntry Result;
  size_t Size = MinimizedFileContents.size();
  Result.MaybeStat = llvm::vfs::Status(Stat->getName(), Stat->getUniqueID(),
                                       Stat->getLastModificationTime(),
                                       Stat->getUser(), Stat->getGroup(), Size,
                                       Stat->getType(), Stat->getPermissions());
  // The contents produced by the minimizer must be null terminated.
  assert(MinimizedFileContents.data()[MinimizedFileContents.size()] == '\0' &&
         "not null terminated contents");
  // Even though there's an implicit null terminator in the minimized contents,
  // we want to temporarily make it explicit. This will ensure that the
  // std::move will preserve it even if it needs to do a copy if the
  // SmallString still has the small capacity.
  MinimizedFileContents.push_back('\0');
  Result.Contents = std::move(MinimizedFileContents);
  // Now make the null terminator implicit again, so that Clang's lexer can find
  // it right where the buffer ends.
  Result.Contents.pop_back();

  // Compute the skipped PP ranges that speedup skipping over inactive
  // preprocessor blocks.
  llvm::SmallVector<minimize_source_to_dependency_directives::SkippedRange, 32>
      SkippedRanges;
  minimize_source_to_dependency_directives::computeSkippedRanges(Tokens,
                                                                 SkippedRanges);
  PreprocessorSkippedRangeMapping Mapping;
  for (const auto &Range : SkippedRanges) {
    if (Range.Length < 16) {
      // Ignore small ranges as non-profitable.
      // FIXME: This is a heuristic, its worth investigating the tradeoffs
      // when it should be applied.
      continue;
    }
    Mapping[Range.Offset] = Range.Length;
  }
  Result.PPSkippedRangeMapping = std::move(Mapping);

  return Result;
}

CachedFileSystemEntry
CachedFileSystemEntry::createDirectoryEntry(llvm::vfs::Status &&Stat) {
  assert(Stat.isDirectory() && "not a directory!");
  auto Result = CachedFileSystemEntry();
  Result.MaybeStat = std::move(Stat);
  return Result;
}

DependencyScanningFilesystemSharedCache::
    DependencyScanningFilesystemSharedCache() {
  // This heuristic was chosen using a empirical testing on a
  // reasonably high core machine (iMacPro 18 cores / 36 threads). The cache
  // sharding gives a performance edge by reducing the lock contention.
  // FIXME: A better heuristic might also consider the OS to account for
  // the different cost of lock contention on different OSes.
  NumShards = std::max(2u, llvm::hardware_concurrency() / 4);
  CacheShards = std::make_unique<CacheShard[]>(NumShards);
}

/// Returns a cache entry for the corresponding key.
///
/// A new cache entry is created if the key is not in the cache. This is a
/// thread safe call.
DependencyScanningFilesystemSharedCache::SharedFileSystemEntry &
DependencyScanningFilesystemSharedCache::get(StringRef Key) {
  CacheShard &Shard = CacheShards[llvm::hash_value(Key) % NumShards];
  std::unique_lock<std::mutex> LockGuard(Shard.CacheLock);
  auto It = Shard.Cache.try_emplace(Key);
  return It.first->getValue();
}

/// Whitelist file extensions that should be minimized, treating no extension as
/// a source file that should be minimized.
///
/// This is kinda hacky, it would be better if we knew what kind of file Clang
/// was expecting instead.
static bool shouldMinimize(StringRef Filename) {
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
  return shouldMinimize(Filename); // Only cache stat failures on source files.
}

llvm::ErrorOr<const CachedFileSystemEntry *>
DependencyScanningWorkerFilesystem::getOrCreateFileSystemEntry(
    const StringRef Filename) {
  if (const CachedFileSystemEntry *Entry = getCachedEntry(Filename)) {
    return Entry;
  }

  // FIXME: Handle PCM/PCH files.
  // FIXME: Handle module map files.

  bool KeepOriginalSource = IgnoredFiles.count(Filename) ||
                            !shouldMinimize(Filename);
  DependencyScanningFilesystemSharedCache::SharedFileSystemEntry
      &SharedCacheEntry = SharedCache.get(Filename);
  const CachedFileSystemEntry *Result;
  {
    std::unique_lock<std::mutex> LockGuard(SharedCacheEntry.ValueLock);
    CachedFileSystemEntry &CacheEntry = SharedCacheEntry.Value;

    if (!CacheEntry.isValid()) {
      llvm::vfs::FileSystem &FS = getUnderlyingFS();
      auto MaybeStatus = FS.status(Filename);
      if (!MaybeStatus) {
        if (!shouldCacheStatFailures(Filename))
          // HACK: We need to always restat non source files if the stat fails.
          //   This is because Clang first looks up the module cache and module
          //   files before building them, and then looks for them again. If we
          //   cache the stat failure, it won't see them the second time.
          return MaybeStatus.getError();
        else
          CacheEntry = CachedFileSystemEntry(MaybeStatus.getError());
      } else if (MaybeStatus->isDirectory())
        CacheEntry = CachedFileSystemEntry::createDirectoryEntry(
            std::move(*MaybeStatus));
      else
        CacheEntry = CachedFileSystemEntry::createFileEntry(
            Filename, FS, !KeepOriginalSource);
    }

    Result = &CacheEntry;
  }

  // Store the result in the local cache.
  setCachedEntry(Filename, Result);
  return Result;
}

llvm::ErrorOr<llvm::vfs::Status>
DependencyScanningWorkerFilesystem::status(const Twine &Path) {
  SmallString<256> OwnedFilename;
  StringRef Filename = Path.toStringRef(OwnedFilename);
  const llvm::ErrorOr<const CachedFileSystemEntry *> Result =
      getOrCreateFileSystemEntry(Filename);
  if (!Result)
    return Result.getError();
  return (*Result)->getStatus();
}

namespace {

/// The VFS that is used by clang consumes the \c CachedFileSystemEntry using
/// this subclass.
class MinimizedVFSFile final : public llvm::vfs::File {
public:
  MinimizedVFSFile(std::unique_ptr<llvm::MemoryBuffer> Buffer,
                   llvm::vfs::Status Stat)
      : Buffer(std::move(Buffer)), Stat(std::move(Stat)) {}

  llvm::ErrorOr<llvm::vfs::Status> status() override { return Stat; }

  const llvm::MemoryBuffer *getBufferPtr() const { return Buffer.get(); }

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

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
createFile(const CachedFileSystemEntry *Entry,
           ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings) {
  if (Entry->isDirectory())
    return llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>(
        std::make_error_code(std::errc::is_a_directory));
  llvm::ErrorOr<StringRef> Contents = Entry->getContents();
  if (!Contents)
    return Contents.getError();
  auto Result = std::make_unique<MinimizedVFSFile>(
      llvm::MemoryBuffer::getMemBuffer(*Contents, Entry->getName(),
                                       /*RequiresNullTerminator=*/false),
      *Entry->getStatus());
  if (!Entry->getPPSkippedRangeMapping().empty() && PPSkipMappings)
    (*PPSkipMappings)[Result->getBufferPtr()] =
        &Entry->getPPSkippedRangeMapping();
  return llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>(
      std::unique_ptr<llvm::vfs::File>(std::move(Result)));
}

} // end anonymous namespace

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
DependencyScanningWorkerFilesystem::openFileForRead(const Twine &Path) {
  SmallString<256> OwnedFilename;
  StringRef Filename = Path.toStringRef(OwnedFilename);

  const llvm::ErrorOr<const CachedFileSystemEntry *> Result =
      getOrCreateFileSystemEntry(Filename);
  if (!Result)
    return Result.getError();
  return createFile(Result.get(), PPSkipMappings);
}
