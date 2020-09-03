//===--- BackgroundIndexRebuild.h - when to rebuild the bg index--*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation detail of the background indexer
// (Background.h), which is exposed for testing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUND_INDEX_REBUILD_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUND_INDEX_REBUILD_H

#include "index/FileIndex.h"
#include "index/Index.h"
#include "llvm/Support/Threading.h"
#include <cstddef>

namespace clang {
namespace clangd {

// The BackgroundIndexRebuilder builds the serving data structures periodically
// in response to events in the background indexer. The goal is to ensure the
// served data stays fairly fresh, without wasting lots of CPU rebuilding it
// often.
//
// The index is always built after a set of shards are loaded from disk.
// This happens when clangd discovers a compilation database that we've
// previously built an index for. It's a fairly fast process that yields lots
// of data, so we wait to get all of it.
//
// The index is built after indexing a few translation units, if it wasn't built
// already. This ensures quick startup if there's no existing index.
// Waiting for a few random TUs yields coverage of the most common headers.
//
// The index is rebuilt every N TUs, to keep if fresh as files are indexed.
//
// The index is rebuilt every time the queue goes idle, if it's stale.
//
// All methods are threadsafe. They're called after FileSymbols is updated
// etc. Without external locking, the rebuilt index may include more updates
// than intended, which is fine.
//
// This class is exposed in the header so it can be tested.
class BackgroundIndexRebuilder {
public:
  BackgroundIndexRebuilder(SwapIndex *Target, FileSymbols *Source,
                           unsigned Threads)
      : TUsBeforeFirstBuild(Threads), Target(Target), Source(Source) {}

  // Called to indicate a TU has been indexed.
  // May rebuild, if enough TUs have been indexed.
  void indexedTU();
  // Called to indicate that all worker threads are idle.
  // May reindex, if the index is not up to date.
  void idle();
  // Called to indicate we're going to load a batch of shards from disk.
  // startLoading() and doneLoading() must be paired, but multiple loading
  // sessions may happen concurrently.
  void startLoading();
  // Called to indicate some shards were actually loaded from disk.
  void loadedShard(size_t ShardCount);
  // Called to indicate we're finished loading shards from disk.
  // May rebuild (if any were loaded).
  void doneLoading();

  // Ensures we won't start any more rebuilds.
  void shutdown();

  // Thresholds for rebuilding as TUs get indexed. Exposed for testing.
  const unsigned TUsBeforeFirstBuild; // Typically one per worker thread.
  const unsigned TUsBeforeRebuild = 100;

private:
  // Run Check under the lock, and rebuild if it returns true.
  void maybeRebuild(const char *Reason, std::function<bool()> Check);
  bool enoughTUsToRebuild() const;

  // All transient state is guarded by the mutex.
  std::mutex Mu;
  bool ShouldStop = false;
  // Index builds are versioned. ActiveVersion chases StartedVersion.
  unsigned StartedVersion = 0;
  unsigned ActiveVersion = 0;
  // How many TUs have we indexed so far since startup?
  unsigned IndexedTUs = 0;
  unsigned IndexedTUsAtLastRebuild = 0;
  // Are we loading shards? May be multiple concurrent sessions.
  unsigned Loading = 0;
  unsigned LoadedShards; // In the current loading session.

  SwapIndex *Target;
  FileSymbols *Source;
};

} // namespace clangd
} // namespace clang

#endif
