//===--- Trigram.h - Trigram generation for Fuzzy Matching ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TRIGRAM_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TRIGRAM_H

#include "Token.h"

#include <string>

namespace clang {
namespace clangd {
namespace dex {

/// Returns list of unique fuzzy-search trigrams from unqualified symbol.
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
/// Trigrams in the returned list are deduplicated.
std::vector<Token> generateIdentifierTrigrams(llvm::StringRef Identifier);

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

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TRIGRAM_H
