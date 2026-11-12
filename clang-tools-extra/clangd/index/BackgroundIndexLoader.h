//===--- BackgroundIndexLoader.h - Load shards from index storage-*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUNDINDEXLOADER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_BACKGROUNDINDEXLOADER_H

#include "index/Background.h"
#include "support/Path.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <vector>

namespace clang {
namespace clangd {

/// Represents a shard loaded from storage, stores contents in \p Shard and
/// metadata about the source file that generated this shard.
struct LoadedShard {
  /// Path of the source file that produced this shard.
  Path AbsolutePath;
  /// Digest of the source file contents that produced this shard.
  FileDigest Digest = {};
  /// Whether the RefSlab in Shard should be used for updating symbol reference
  /// counts when building an index.
  bool CountReferences = false;
  /// Whether the indexing action producing that shard had errors.
  bool HadErrors = false;
  /// Path to a TU that is depending on this shard.
  Path DependentTU;
  /// Will be nullptr when index storage couldn't provide a valid shard for
  /// AbsolutePath.
  std::unique_ptr<IndexFileIn> Shard;
};

/// Loads all shards for the TU \p MainFile from \p Storage.
std::vector<LoadedShard>
loadIndexShards(llvm::ArrayRef<Path> MainFiles,
                BackgroundIndexStorage::Factory &IndexStorageFactory,
                const GlobalCompilationDatabase &CDB);

} // namespace clangd
} // namespace clang

#endif
