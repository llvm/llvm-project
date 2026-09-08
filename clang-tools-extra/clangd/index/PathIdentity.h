//===--- PathIdentity.h - File identity for index keys ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_PATHIDENTITY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_PATHIDENTITY_H

#include "support/Path.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace clang {
namespace clangd {

/// Filesystem paths use Path identity. Other URI schemes are opaque and
/// case-sensitive: resolving them can require a workspace hint we don't have.
struct IndexFileKeyRef {
  llvm::StringRef Value;
  enum Kind { FilePath, OpaqueURI } K = FilePath;

  friend bool operator==(IndexFileKeyRef L, IndexFileKeyRef R) {
    return L.K == R.K && (L.K == FilePath ? pathEquals(L.Value, R.Value)
                                          : L.Value == R.Value);
  }
};

class IndexFileKey {
public:
  explicit IndexFileKey(IndexFileKeyRef Ref)
      : Value(Ref.Value.str()), K(Ref.K) {}
  operator IndexFileKeyRef() const { return {Value, K}; }
  llvm::StringRef raw() const { return Value; }
  friend bool operator==(const IndexFileKey &L, const IndexFileKey &R) {
    return IndexFileKeyRef(L) == IndexFileKeyRef(R);
  }
  friend bool operator!=(const IndexFileKey &L, const IndexFileKey &R) {
    return !(L == R);
  }

private:
  std::string Value;
  IndexFileKeyRef::Kind K;
};

struct IndexFileKeyInfo {
  static unsigned getHashValue(IndexFileKeyRef Key) {
    return Key.K == IndexFileKeyRef::FilePath
               ? pathHash(Key.Value)
               : llvm::DenseMapInfo<llvm::StringRef>::getHashValue(Key.Value);
  }
  static bool isEqual(IndexFileKeyRef L, IndexFileKeyRef R) { return L == R; }
};

template <typename T>
using IndexFileMap = llvm::DenseMap<IndexFileKey, T, IndexFileKeyInfo>;
using IndexFileSet = llvm::DenseSet<IndexFileKey, IndexFileKeyInfo>;

/// Invalid file URIs are diagnosed and rejected, never treated as paths.
std::optional<IndexFileKey> indexFileIdentity(llvm::StringRef URIOrPath);
/// Borrowed lookup key, backed by URIOrPath or Storage. Common unescaped file
/// URIs and opaque URIs need no allocation or scheme resolution.
std::optional<IndexFileKeyRef>
indexFileIdentity(llvm::StringRef URIOrPath,
                  llvm::SmallVectorImpl<char> &Storage);

inline std::optional<IndexFileKey> indexFileIdentityFrom(llvm::StringRef S) {
  return indexFileIdentity(S);
}
inline std::optional<IndexFileKey> indexFileIdentityFrom(PathRef P) {
  return IndexFileKey({P.raw(), IndexFileKeyRef::FilePath});
}
inline std::optional<IndexFileKey> indexFileIdentityFrom(const Path &P) {
  return indexFileIdentityFrom(P.ref());
}
inline std::optional<IndexFileKey>
indexFileIdentityFrom(const IndexFileKey &K) {
  return K;
}
template <typename Val>
inline std::optional<IndexFileKey>
indexFileIdentityFrom(const llvm::StringMapEntry<Val> &E) {
  return indexFileIdentity(E.getKey());
}

/// Transfer already-normalized sets without copying their keys or rehashing.
inline IndexFileSet indexFileIdentities(IndexFileSet Files) { return Files; }

template <typename FileRange>
IndexFileSet indexFileIdentities(const FileRange &Files) {
  IndexFileSet Result;
  for (const auto &File : Files)
    if (auto Identity = indexFileIdentityFrom(File))
      Result.insert(std::move(*Identity));
  return Result;
}

} // namespace clangd
} // namespace clang

#endif
