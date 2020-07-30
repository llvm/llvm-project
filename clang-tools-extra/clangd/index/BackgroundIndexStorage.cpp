//== BackgroundIndexStorage.cpp - Provide caching support to BackgroundIndex ==/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "index/Background.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <functional>

namespace clang {
namespace clangd {
namespace {

std::string getShardPathFromFilePath(llvm::StringRef ShardRoot,
                                     llvm::StringRef FilePath) {
  llvm::SmallString<128> ShardRootSS(ShardRoot);
  llvm::sys::path::append(ShardRootSS, llvm::sys::path::filename(FilePath) +
                                           "." + llvm::toHex(digest(FilePath)) +
                                           ".idx");
  return std::string(ShardRootSS.str());
}

// Uses disk as a storage for index shards.
class DiskBackedIndexStorage : public BackgroundIndexStorage {
  std::string DiskShardRoot;

public:
  // Creates `DiskShardRoot` and any parents during construction.
  DiskBackedIndexStorage(llvm::StringRef Directory) : DiskShardRoot(Directory) {
    std::error_code OK;
    std::error_code EC = llvm::sys::fs::create_directories(DiskShardRoot);
    if (EC != OK) {
      elog("Failed to create directory {0} for index storage: {1}",
           DiskShardRoot, EC.message());
    }
  }

  std::unique_ptr<IndexFileIn>
  loadShard(llvm::StringRef ShardIdentifier) const override {
    const std::string ShardPath =
        getShardPathFromFilePath(DiskShardRoot, ShardIdentifier);
    auto Buffer = llvm::MemoryBuffer::getFile(ShardPath);
    if (!Buffer)
      return nullptr;
    if (auto I = readIndexFile(Buffer->get()->getBuffer()))
      return std::make_unique<IndexFileIn>(std::move(*I));
    else
      elog("Error while reading shard {0}: {1}", ShardIdentifier,
           I.takeError());
    return nullptr;
  }

  llvm::Error storeShard(llvm::StringRef ShardIdentifier,
                         IndexFileOut Shard) const override {
    auto ShardPath = getShardPathFromFilePath(DiskShardRoot, ShardIdentifier);
    return llvm::writeFileAtomically(ShardPath + ".tmp.%%%%%%%%", ShardPath,
                                     [&Shard](llvm::raw_ostream &OS) {
                                       OS << Shard;
                                       return llvm::Error::success();
                                     });
  }
};

// Doesn't persist index shards anywhere (used when the CDB dir is unknown).
// We could consider indexing into ~/.clangd/ or so instead.
class NullStorage : public BackgroundIndexStorage {
public:
  std::unique_ptr<IndexFileIn>
  loadShard(llvm::StringRef ShardIdentifier) const override {
    return nullptr;
  }

  llvm::Error storeShard(llvm::StringRef ShardIdentifier,
                         IndexFileOut Shard) const override {
    vlog("Couldn't find project for {0}, indexing in-memory only",
         ShardIdentifier);
    return llvm::Error::success();
  }
};

// Creates and owns IndexStorages for multiple CDBs.
// When a CDB root is found, shards are stored in $ROOT/.cache/clangd/index/.
// When no root is found, the fallback path is ~/.cache/clangd/index/.
class DiskBackedIndexStorageManager {
public:
  DiskBackedIndexStorageManager(
      std::function<llvm::Optional<ProjectInfo>(PathRef)> GetProjectInfo)
      : IndexStorageMapMu(std::make_unique<std::mutex>()),
        GetProjectInfo(std::move(GetProjectInfo)) {
    llvm::SmallString<128> FallbackDir;
    if (llvm::sys::path::cache_directory(FallbackDir))
      llvm::sys::path::append(FallbackDir, "clangd", "index");
    this->FallbackDir = FallbackDir.str().str();
  }

  // Creates or fetches to storage from cache for the specified project.
  BackgroundIndexStorage *operator()(PathRef File) {
    std::lock_guard<std::mutex> Lock(*IndexStorageMapMu);
    llvm::SmallString<128> StorageDir(FallbackDir);
    if (auto PI = GetProjectInfo(File)) {
      StorageDir = PI->SourceRoot;
      llvm::sys::path::append(StorageDir, ".cache", "clangd", "index");
    }
    auto &IndexStorage = IndexStorageMap[StorageDir];
    if (!IndexStorage)
      IndexStorage = create(StorageDir);
    return IndexStorage.get();
  }

private:
  std::unique_ptr<BackgroundIndexStorage> create(PathRef CDBDirectory) {
    if (CDBDirectory.empty()) {
      elog("Tried to create storage for empty directory!");
      return std::make_unique<NullStorage>();
    }
    return std::make_unique<DiskBackedIndexStorage>(CDBDirectory);
  }

  Path FallbackDir;

  llvm::StringMap<std::unique_ptr<BackgroundIndexStorage>> IndexStorageMap;
  std::unique_ptr<std::mutex> IndexStorageMapMu;

  std::function<llvm::Optional<ProjectInfo>(PathRef)> GetProjectInfo;
};

} // namespace

BackgroundIndexStorage::Factory
BackgroundIndexStorage::createDiskBackedStorageFactory(
    std::function<llvm::Optional<ProjectInfo>(PathRef)> GetProjectInfo) {
  return DiskBackedIndexStorageManager(std::move(GetProjectInfo));
}

} // namespace clangd
} // namespace clang
