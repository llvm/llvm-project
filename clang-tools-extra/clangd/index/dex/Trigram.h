//===--- Trigram.h - Trigram generation for Fuzzy Matching ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Trigrams are attributes of the symbol unqualified name used to effectively
/// extract symbols which can be fuzzy-matched given user query from the
/// inverted index. To match query with the extracted set of trigrams Q, the set
/// of generated trigrams T for identifier (unqualified symbol name) should
/// contain all items of Q, i.e. Q ⊆ T.
///
/// Trigram sets extracted from unqualified name and from query are different:
/// the set of query trigrams only contains consecutive sequences of three
/// characters (which is only a subset of all trigrams generated for an
/// identifier).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_TRIGRAM_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_TRIGRAM_H

#include "index/dex/Token.h"
#include "llvm/ADT/bit.h"

#include <array>
#include <string>

namespace clang {
namespace clangd {
namespace dex {

// Compact representation of a trigram (string with up to 3 characters).
// Trigram generation is the hot path of indexing, so Token is too wasteful.
class Trigram {
  std::array<char, 4> Data; // Last element is length.
  // Steal some invalid bit patterns for DenseMap sentinels.
  enum class Sentinel { Tombstone = 4, Empty = 5 };
  Trigram(Sentinel S) : Data{0, 0, 0, static_cast<char>(S)} {}
  uint32_t id() const { return llvm::bit_cast<uint32_t>(Data); }

public:
  Trigram() : Data{0, 0, 0, 0} {}
  Trigram(char A) : Data{A, 0, 0, 1} {}
  Trigram(char A, char B) : Data{A, B, 0, 2} {}
  Trigram(char A, char B, char C) : Data{A, B, C, 3} {}
  std::string str() const { return std::string(Data.data(), Data[3]); }
  friend struct ::llvm::DenseMapInfo<Trigram>;
  friend bool operator==(Trigram L, Trigram R) { return L.id() == R.id(); }
  friend bool operator<(Trigram L, Trigram R) { return L.id() < R.id(); }
};

/// Produces list of unique fuzzy-search trigrams from unqualified symbol.
/// The trigrams give the 3-character query substrings this symbol can match.
///
/// The symbol's name is broken into segments, e.g. "FooBar" has two segments.
/// Trigrams can start at any character in the input. Then we can choose to move
/// to the next character or to the start of the next segment.
///
/// Short trigrams (length 1-2) are used for short queries. These are:
///  - prefixes of the identifier, of length 1 and 2
///  - the first character + next head character
///
/// For "FooBar" we get the following trigrams:
///  {f, fo, fb, foo, fob, fba, oob, oba, bar}.
///
/// Trigrams are lowercase, as trigram matching is case-insensitive.
/// Trigrams in the list are deduplicated.
void generateIdentifierTrigrams(llvm::StringRef Identifier,
                                std::vector<Trigram> &Out);

/// Returns list of unique fuzzy-search trigrams given a query.
///
/// Query is segmented using FuzzyMatch API and downcasted to lowercase. Then,
/// the simplest trigrams - sequences of three consecutive letters and digits
/// are extracted and returned after deduplication.
///
/// For short queries (less than 3 characters with Head or Tail roles in Fuzzy
/// Matching segmentation) this returns a single trigram with the first
/// characters (up to 3) to perform prefix match.
std::vector<Token> generateQueryTrigrams(llvm::StringRef Query);

} // namespace dex
} // namespace clangd
} // namespace clang

namespace llvm {
template <> struct DenseMapInfo<clang::clangd::dex::Trigram> {
  using Trigram = clang::clangd::dex::Trigram;
  static inline Trigram getEmptyKey() {
    return Trigram(Trigram::Sentinel::Empty);
  }
  static inline Trigram getTombstoneKey() {
    return Trigram(Trigram::Sentinel::Tombstone);
  }
  static unsigned getHashValue(Trigram V) {
    // Finalize step from MurmurHash3.
    uint32_t X = V.id();
    X ^= X >> 16;
    X *= uint32_t{0x85ebca6b};
    X ^= X >> 13;
    X *= uint32_t{0xc2b2ae35};
    X ^= X >> 16;
    return X;
  }
  static bool isEqual(const Trigram &LHS, const Trigram &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_TRIGRAM_H
