//===-- BackgroundIndexLoader.cpp - ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/BackgroundIndexLoader.h"
#include "GlobalCompilationDatabase.h"
#include "index/Background.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

/// A helper class to cache BackgroundIndexStorage operations and keep the
/// inverse dependency mapping.
class BackgroundIndexLoader {
public:
  BackgroundIndexLoader(BackgroundIndexStorage::Factory &IndexStorageFactory)
      : IndexStorageFactory(IndexStorageFactory) {}
  /// Load the shards for \p MainFile and all of its dependencies.
  void load(PathRef MainFile);

  /// Consumes the loader and returns all shards.
  std::vector<LoadedShard> takeResult() &&;

private:
  /// Returns the Shard for \p StartSourceFile from cache or loads it from \p
  /// Storage. Also returns paths for dependencies of \p StartSourceFile if it
  /// wasn't cached yet.
  std::pair<const LoadedShard &, std::vector<Path>>
  loadShard(PathRef StartSourceFile, PathRef DependentTU);

  /// Cache for Storage lookups.
  llvm::StringMap<LoadedShard> LoadedShards;

  BackgroundIndexStorage::Factory &IndexStorageFactory;
};

std::pair<const LoadedShard &, std::vector<Path>>
BackgroundIndexLoader::loadShard(PathRef StartSourceFile, PathRef DependentTU) {
  auto It = LoadedShards.try_emplace(StartSourceFile);
  LoadedShard &LS = It.first->getValue();
  std::vector<Path> Edges = {};
  // Return the cached shard.
  if (!It.second)
    return {LS, Edges};

  LS.AbsolutePath = StartSourceFile.str();
  LS.DependentTU = std::string(DependentTU);
  BackgroundIndexStorage *Storage = IndexStorageFactory(LS.AbsolutePath);
  auto Shard = Storage->loadShard(StartSourceFile);
  if (!Shard || !Shard->Sources) {
    vlog("Failed to load shard: {0}", StartSourceFile);
    return {LS, Edges};
  }

  LS.Shard = std::move(Shard);
  for (const auto &It : *LS.Shard->Sources) {
    auto AbsPath = URI::resolve(It.getKey(), StartSourceFile);
    if (!AbsPath) {
      elog("Failed to resolve URI: {0}", AbsPath.takeError());
      continue;
    }
    // A shard contains only edges for non main-file sources.
    if (*AbsPath != StartSourceFile) {
      Edges.push_back(*AbsPath);
      continue;
    }

    // Fill in shard metadata.
    const IncludeGraphNode &IGN = It.getValue();
    LS.Digest = IGN.Digest;
    LS.CountReferences = IGN.Flags & IncludeGraphNode::SourceFlag::IsTU;
    LS.HadErrors = IGN.Flags & IncludeGraphNode::SourceFlag::HadErrors;
  }
  assert(LS.Digest != FileDigest{{0}} && "Digest is empty?");
  return {LS, Edges};
}

void BackgroundIndexLoader::load(PathRef MainFile) {
  llvm::StringSet<> InQueue;
  // Following containers points to strings inside InQueue.
  std::queue<PathRef> ToVisit;
  InQueue.insert(MainFile);
  ToVisit.push(MainFile);

  while (!ToVisit.empty()) {
    PathRef SourceFile = ToVisit.front();
    ToVisit.pop();

    auto ShardAndEdges = loadShard(SourceFile, MainFile);
    for (PathRef Edge : ShardAndEdges.second) {
      auto It = InQueue.insert(Edge);
      if (It.second)
        ToVisit.push(It.first->getKey());
    }
  }
}

std::vector<LoadedShard> BackgroundIndexLoader::takeResult() && {
  std::vector<LoadedShard> Result;
  Result.reserve(LoadedShards.size());
  for (auto &It : LoadedShards)
    Result.push_back(std::move(It.getValue()));
  return Result;
}
} // namespace

std::vector<LoadedShard>
loadIndexShards(llvm::ArrayRef<Path> MainFiles,
                BackgroundIndexStorage::Factory &IndexStorageFactory,
                const GlobalCompilationDatabase &CDB) {
  BackgroundIndexLoader Loader(IndexStorageFactory);
  for (llvm::StringRef MainFile : MainFiles) {
    assert(llvm::sys::path::is_absolute(MainFile));
    Loader.load(MainFile);
  }
  return std::move(Loader).takeResult();
}

} // namespace clangd
} // namespace clang
