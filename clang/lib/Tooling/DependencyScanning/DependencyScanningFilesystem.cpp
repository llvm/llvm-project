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
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/Threading.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

llvm::ErrorOr<llvm::vfs::Status>
CachedFileSystemEntry::initFile(StringRef Filename, llvm::vfs::FileSystem &FS) {
  // Load the file and its content from the file system.
  auto MaybeFile = FS.openFileForRead(Filename);
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

  OriginalContents = std::move(Buffer);
  return Stat;
}

void CachedFileSystemEntry::minimizeFile() {
  assert(OriginalContents && "minimizing missing contents");

  llvm::SmallString<1024> MinimizedFileContents;
  // Minimize the file down to directives that might affect the dependencies.
  SmallVector<minimize_source_to_dependency_directives::Token, 64> Tokens;
  if (minimizeSourceToDependencyDirectives(OriginalContents->getBuffer(),
                                           MinimizedFileContents, Tokens)) {
    // FIXME: Propagate the diagnostic if desired by the client.
    // Use the original file if the minimization failed.
    MinimizedContentsStorage =
        llvm::MemoryBuffer::getMemBuffer(*OriginalContents);
    MinimizedContentsAccess.store(MinimizedContentsStorage.get());
    return;
  }

  // The contents produced by the minimizer must be null terminated.
  assert(MinimizedFileContents.data()[MinimizedFileContents.size()] == '\0' &&
         "not null terminated contents");

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
  PPSkippedRangeMapping = std::move(Mapping);

  MinimizedContentsStorage = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(MinimizedFileContents));
  // The algorithm in `getOrCreateFileSystemEntry` uses the presence of
  // minimized contents to decide whether an entry is up-to-date or not.
  // If it is up-to-date, the skipped range mappings must be already computed.
  // This is why we need to store the minimized contents **after** storing the
  // skipped range mappings. Failing to do so would lead to a data race.
  MinimizedContentsAccess.store(MinimizedContentsStorage.get());
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

DependencyScanningFilesystemSharedCache::SharedFileSystemEntry &
DependencyScanningFilesystemSharedCache::get(StringRef Key) {
  CacheShard &Shard = CacheShards[llvm::hash_value(Key) % NumShards];
  std::lock_guard<std::mutex> LockGuard(Shard.CacheLock);
  auto It = Shard.Cache.try_emplace(Key);
  return It.first->getValue();
}

/// Whitelist file extensions that should be minimized, treating no extension as
/// a source file that should be minimized.
///
/// This is kinda hacky, it would be better if we knew what kind of file Clang
/// was expecting instead.
static bool shouldMinimizeBasedOnExtension(StringRef Filename) {
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
  return shouldMinimizeBasedOnExtension(Filename);
}

void DependencyScanningWorkerFilesystem::disableMinimization(
    StringRef RawFilename) {
  llvm::SmallString<256> Filename;
  llvm::sys::path::native(RawFilename, Filename);
  NotToBeMinimized.insert(Filename);
}

bool DependencyScanningWorkerFilesystem::shouldMinimize(StringRef RawFilename) {
  if (!shouldMinimizeBasedOnExtension(RawFilename))
    return false;

  llvm::SmallString<256> Filename;
  llvm::sys::path::native(RawFilename, Filename);
  return !NotToBeMinimized.contains(Filename);
}

void CachedFileSystemEntry::init(llvm::ErrorOr<llvm::vfs::Status> &&MaybeStatus,
                                 StringRef Filename,
                                 llvm::vfs::FileSystem &FS) {
  if (!MaybeStatus || MaybeStatus->isDirectory())
    MaybeStat = std::move(MaybeStatus);
  else
    MaybeStat = initFile(Filename, FS);
}

llvm::ErrorOr<EntryRef>
DependencyScanningWorkerFilesystem::getOrCreateFileSystemEntry(
    StringRef Filename) {
  bool ShouldBeMinimized = shouldMinimize(Filename);

  const auto *Entry = LocalCache.getCachedEntry(Filename);
  if (Entry && !Entry->needsUpdate(ShouldBeMinimized))
    return EntryRef(ShouldBeMinimized, *Entry);

  // FIXME: Handle PCM/PCH files.
  // FIXME: Handle module map files.

  auto &SharedCacheEntry = SharedCache.get(Filename);
  {
    std::lock_guard<std::mutex> LockGuard(SharedCacheEntry.ValueLock);
    CachedFileSystemEntry &CacheEntry = SharedCacheEntry.Value;

    if (!CacheEntry.isInitialized()) {
      auto MaybeStatus = getUnderlyingFS().status(Filename);
      if (!MaybeStatus && !shouldCacheStatFailures(Filename))
        // HACK: We need to always restat non source files if the stat fails.
        //   This is because Clang first looks up the module cache and module
        //   files before building them, and then looks for them again. If we
        //   cache the stat failure, it won't see them the second time.
        return MaybeStatus.getError();
      CacheEntry.init(std::move(MaybeStatus), Filename, getUnderlyingFS());
    }

    // Checking `needsUpdate` verifies the entry represents an opened file.
    // Only checking `needsMinimization` could lead to minimization of files
    // that we failed to load (such files don't have `OriginalContents`).
    if (CacheEntry.needsUpdate(ShouldBeMinimized))
      CacheEntry.minimizeFile();
  }

  // Store the result in the local cache.
  Entry = &SharedCacheEntry.Value;
  return EntryRef(ShouldBeMinimized, *Entry);
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
class MinimizedVFSFile final : public llvm::vfs::File {
public:
  MinimizedVFSFile(std::unique_ptr<llvm::MemoryBuffer> Buffer,
                   llvm::vfs::Status Stat)
      : Buffer(std::move(Buffer)), Stat(std::move(Stat)) {}

  static llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  create(EntryRef Entry,
         ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings);

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

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> MinimizedVFSFile::create(
    EntryRef Entry, ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings) {
  if (Entry.isDirectory())
    return std::make_error_code(std::errc::is_a_directory);

  llvm::ErrorOr<StringRef> Contents = Entry.getContents();
  if (!Contents)
    return Contents.getError();
  auto Result = std::make_unique<MinimizedVFSFile>(
      llvm::MemoryBuffer::getMemBuffer(*Contents, Entry.getName(),
                                       /*RequiresNullTerminator=*/false),
      *Entry.getStatus());

  const auto *EntrySkipMappings = Entry.getPPSkippedRangeMapping();
  if (EntrySkipMappings && !EntrySkipMappings->empty() && PPSkipMappings)
    (*PPSkipMappings)[Result->Buffer->getBufferStart()] = EntrySkipMappings;

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
  return MinimizedVFSFile::create(Result.get(), PPSkipMappings);
}
