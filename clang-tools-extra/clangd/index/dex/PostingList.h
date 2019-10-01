//===--- PostingList.h - Symbol identifiers storage interface  --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This defines posting list interface: a storage for identifiers of symbols
/// which can be characterized by a specific feature (such as fuzzy-find
/// trigram, scope, type or any other Search Token). Posting lists can be
/// traversed in order using an iterator and are values for inverted index,
/// which maps search tokens to corresponding posting lists.
///
/// In order to decrease size of Index in-memory representation, Variable Byte
/// Encoding (VByte) is used for PostingLists compression. An overview of VByte
/// algorithm can be found in "Introduction to Information Retrieval" book:
/// https://nlp.stanford.edu/IR-book/html/htmledition/variable-byte-codes-1.html
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_POSTINGLIST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_POSTINGLIST_H

#include "Iterator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {
struct Token;

/// NOTE: This is an implementation detail.
///
/// Chunk is a fixed-width piece of PostingList which contains the first DocID
/// in uncompressed format (Head) and delta-encoded Payload. It can be
/// decompressed upon request.
struct Chunk {
  /// Keep sizeof(Chunk) == 32.
  static constexpr size_t PayloadSize = 32 - sizeof(DocID);

  llvm::SmallVector<DocID, PayloadSize + 1> decompress() const;

  /// The first element of decompressed Chunk.
  DocID Head;
  /// VByte-encoded deltas.
  std::array<uint8_t, PayloadSize> Payload;
};
static_assert(sizeof(Chunk) == 32, "Chunk should take 32 bytes of memory.");

/// PostingList is the storage of DocIDs which can be inserted to the Query
/// Tree as a leaf by constructing Iterator over the PostingList object. DocIDs
/// are stored in underlying chunks. Compression saves memory at a small cost
/// in access time, which is still fast enough in practice.
class PostingList {
public:
  explicit PostingList(llvm::ArrayRef<DocID> Documents);

  /// Constructs DocumentIterator over given posting list. DocumentIterator will
  /// go through the chunks and decompress them on-the-fly when necessary.
  /// If given, Tok is only used for the string representation.
  std::unique_ptr<Iterator> iterator(const Token *Tok = nullptr) const;

  /// Returns in-memory size of external storage.
  size_t bytes() const { return Chunks.capacity() * sizeof(Chunk); }

private:
  const std::vector<Chunk> Chunks;
};

} // namespace dex
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_POSTINGLIST_H
