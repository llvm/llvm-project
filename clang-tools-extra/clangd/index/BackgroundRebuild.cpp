//===-- BackgroundRebuild.cpp - when to rebuild the background index ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/BackgroundRebuild.h"
#include "index/FileIndex.h"
#include "support/Logger.h"
#include "support/Trace.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <thread>

namespace clang {
namespace clangd {

bool BackgroundIndexRebuilder::enoughTUsToRebuild() const {
  if (!ActiveVersion)                         // never built
    return IndexedTUs == TUsBeforeFirstBuild; // use low threshold
  // rebuild if we've reached the (higher) threshold
  return IndexedTUs >= IndexedTUsAtLastRebuild + TUsBeforeRebuild;
}

void BackgroundIndexRebuilder::indexedTU() {
  maybeRebuild("after indexing enough files", [this] {
    ++IndexedTUs;
    if (Loading)
      return false;                      // rebuild once loading finishes
    if (ActiveVersion != StartedVersion) // currently building
      return false;                      // no urgency, avoid overlapping builds
    return enoughTUsToRebuild();
  });
}

void BackgroundIndexRebuilder::idle() {
  maybeRebuild("when background indexer is idle", [this] {
    // rebuild if there's anything new in the index.
    // (even if currently rebuilding! this ensures eventual completeness)
    return IndexedTUs > IndexedTUsAtLastRebuild;
  });
}

void BackgroundIndexRebuilder::startLoading() {
  std::lock_guard<std::mutex> Lock(Mu);
  if (!Loading)
    LoadedShards = 0;
  ++Loading;
}
void BackgroundIndexRebuilder::loadedShard(size_t ShardCount) {
  std::lock_guard<std::mutex> Lock(Mu);
  assert(Loading);
  LoadedShards += ShardCount;
}
void BackgroundIndexRebuilder::doneLoading() {
  maybeRebuild("after loading index from disk", [this] {
    assert(Loading);
    --Loading;
    if (Loading)    // was loading multiple batches concurrently
      return false; // rebuild once the last batch is done.
    // Rebuild if we loaded any shards, or if we stopped an indexedTU rebuild.
    return LoadedShards > 0 || enoughTUsToRebuild();
  });
}

void BackgroundIndexRebuilder::shutdown() {
  std::lock_guard<std::mutex> Lock(Mu);
  ShouldStop = true;
}

void BackgroundIndexRebuilder::maybeRebuild(const char *Reason,
                                            std::function<bool()> Check) {
  unsigned BuildVersion = 0;
  {
    std::lock_guard<std::mutex> Lock(Mu);
    if (!ShouldStop && Check()) {
      BuildVersion = ++StartedVersion;
      IndexedTUsAtLastRebuild = IndexedTUs;
    }
  }
  if (BuildVersion) {
    std::unique_ptr<SymbolIndex> NewIndex;
    {
      vlog("BackgroundIndex: building version {0} {1}", BuildVersion, Reason);
      trace::Span Tracer("RebuildBackgroundIndex");
      SPAN_ATTACH(Tracer, "reason", Reason);
      NewIndex = Source->buildIndex(IndexType::Heavy, DuplicateHandling::Merge);
    }
    {
      std::lock_guard<std::mutex> Lock(Mu);
      // Guard against rebuild finishing in the wrong order.
      if (BuildVersion > ActiveVersion) {
        ActiveVersion = BuildVersion;
        vlog("BackgroundIndex: serving version {0} ({1} bytes)", BuildVersion,
             NewIndex->estimateMemoryUsage());
        Target->reset(std::move(NewIndex));
      }
    }
  }
}

} // namespace clangd
} // namespace clang
