//===--- FS.cpp - File system related utils ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FS.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace clang {
namespace clangd {

PreambleFileStatusCache::PreambleFileStatusCache(llvm::StringRef MainFilePath){
  assert(llvm::sys::path::is_absolute(MainFilePath));
  llvm::SmallString<256> MainFileCanonical(MainFilePath);
  llvm::sys::path::remove_dots(MainFileCanonical, /*remove_dot_dot=*/true);
  this->MainFilePath = std::string(MainFileCanonical.str());
}

void PreambleFileStatusCache::update(const llvm::vfs::FileSystem &FS,
                                     llvm::vfs::Status S) {
  // Canonicalize path for later lookup, which is usually by absolute path.
  llvm::SmallString<32> PathStore(S.getName());
  if (FS.makeAbsolute(PathStore))
    return;
  llvm::sys::path::remove_dots(PathStore, /*remove_dot_dot=*/true);
  // Do not cache status for the main file.
  if (PathStore == MainFilePath)
    return;
  // Stores the latest status in cache as it can change in a preamble build.
  StatCache.insert({PathStore, std::move(S)});
}

llvm::Optional<llvm::vfs::Status>
PreambleFileStatusCache::lookup(llvm::StringRef File) const {
  // Canonicalize to match the cached form.
  // Lookup tends to be first by absolute path, so no need to make absolute.
  llvm::SmallString<256> PathLookup(File);
  llvm::sys::path::remove_dots(PathLookup, /*remove_dot_dot=*/true);

  auto I = StatCache.find(PathLookup);
  if (I != StatCache.end())
    // Returned Status name should always match the requested File.
    return llvm::vfs::Status::copyWithNewName(I->getValue(), File);
  return std::nullopt;
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
PreambleFileStatusCache::getProducingFS(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  // This invalidates old status in cache if files are re-`open()`ed or
  // re-`stat()`ed in case file status has changed during preamble build.
  class CollectFS : public llvm::vfs::ProxyFileSystem {
  public:
    CollectFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
              PreambleFileStatusCache &StatCache)
        : ProxyFileSystem(std::move(FS)), StatCache(StatCache) {}

    llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
    openFileForRead(const llvm::Twine &Path) override {
      auto File = getUnderlyingFS().openFileForRead(Path);
      if (!File || !*File)
        return File;
      // Eagerly stat opened file, as the followup `status` call on the file
      // doesn't necessarily go through this FS. This puts some extra work on
      // preamble build, but it should be worth it as preamble can be reused
      // many times (e.g. code completion) and the repeated status call is
      // likely to be cached in the underlying file system anyway.
      if (auto S = File->get()->status())
        StatCache.update(getUnderlyingFS(), std::move(*S));
      return File;
    }

    llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
      auto S = getUnderlyingFS().status(Path);
      if (S)
        StatCache.update(getUnderlyingFS(), *S);
      return S;
    }

  private:
    PreambleFileStatusCache &StatCache;
  };
  return llvm::IntrusiveRefCntPtr<CollectFS>(
      new CollectFS(std::move(FS), *this));
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
PreambleFileStatusCache::getConsumingFS(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) const {
  class CacheVFS : public llvm::vfs::ProxyFileSystem {
  public:
    CacheVFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
             const PreambleFileStatusCache &StatCache)
        : ProxyFileSystem(std::move(FS)), StatCache(StatCache) {}

    llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
      if (auto S = StatCache.lookup(Path.str()))
        return *S;
      return getUnderlyingFS().status(Path);
    }

  private:
    const PreambleFileStatusCache &StatCache;
  };
  return llvm::IntrusiveRefCntPtr<CacheVFS>(new CacheVFS(std::move(FS), *this));
}

Path removeDots(PathRef File) {
  llvm::SmallString<128> CanonPath(File);
  llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
  return CanonPath.str().str();
}

} // namespace clangd
} // namespace clang
