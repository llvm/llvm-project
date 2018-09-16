//===--- PostingList.h - Symbol identifiers storage interface  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines posting list interface: a storage for identifiers of symbols
// which can be characterized by a specific feature (such as fuzzy-find trigram,
// scope, type or any other Search Token). Posting lists can be traversed in
// order using an iterator and are values for inverted index, which maps search
// tokens to corresponding posting lists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_POSTINGLIST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_POSTINGLIST_H

#include "Iterator.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {

class Iterator;

/// PostingList is the storage of DocIDs which can be inserted to the Query
/// Tree as a leaf by constructing Iterator over the PostingList object.
// FIXME(kbobyrev): Use VByte algorithm to compress underlying data.
class PostingList {
public:
  explicit PostingList(const std::vector<DocID> &&Documents)
      : Documents(std::move(Documents)) {}

  std::unique_ptr<Iterator> iterator() const;

  size_t bytes() const { return Documents.size() * sizeof(DocID); }

private:
  const std::vector<DocID> Documents;
};

} // namespace dex
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_POSTINGLIST_H
