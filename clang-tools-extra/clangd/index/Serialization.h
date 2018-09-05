//===--- Serialization.h - Binary serialization of index data ----*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a compact binary serialization of indexed symbols.
//
// It writes two sections:
//  - a string table (which is compressed)
//  - lists of encoded symbols
//
// The format has a simple versioning scheme: the version is embedded in the
// data and non-current versions are rejected when reading.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_RIFF_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_RIFF_H
#include "Index.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {

// Specifies the contents of an index file to be written.
struct IndexFileOut {
  const SymbolSlab *Symbols;
  // TODO: Support serializing symbol occurrences.
  // TODO: Support serializing Dex posting lists.
};
// Serializes an index file. (This is a RIFF container chunk).
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const IndexFileOut &);

// Holds the contents of an index file that was read.
struct IndexFileIn {
  llvm::Optional<SymbolSlab> Symbols;
};
// Parse an index file. The input must be a RIFF container chunk.
llvm::Expected<IndexFileIn> readIndexFile(llvm::StringRef);

} // namespace clangd
} // namespace clang

#endif
