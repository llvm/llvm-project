//===--- Trigram.h - Trigram generation for Fuzzy Matching ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Trigrams are attributes of the symbol unqualified name used to effectively
// extract symbols which can be fuzzy-matched given user query from the inverted
// index. To match query with the extracted set of trigrams Q, the set of
// generated trigrams T for identifier (unqualified symbol name) should contain
// all items of Q, i.e. Q ⊆ T.
//
// Trigram sets extracted from unqualified name and from query are different:
// the set of query trigrams only contains consecutive sequences of three
// characters (which is only a subset of all trigrams generated for an
// identifier).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TRIGRAM_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DEX_TRIGRAM_H

#include "Token.h"

#include <string>

namespace clang {
namespace clangd {
namespace dex {

/// Returns list of unique fuzzy-search trigrams from unqualified symbol.
///
/// First, given Identifier (unqualified symbol name) is segmented using
/// FuzzyMatch API and lowercased. After segmentation, the following technique
/// is applied for generating trigrams: for each letter or digit in the input
/// string the algorithms looks for the possible next and skip-1-next characters
/// which can be jumped to during fuzzy matching. Each combination of such three
/// characters is inserted into the result.
///
/// Trigrams can start at any character in the input. Then we can choose to move
/// to the next character, move to the start of the next segment, or skip over a
/// segment.
///
/// This also generates incomplete trigrams for short query scenarios:
///  * Empty trigram: "$$$".
///  * Unigram: the first character of the identifier.
///  * Bigrams: a 2-char prefix of the identifier and a bigram of the first two
///    HEAD characters (if they exist).
//
/// Note: the returned list of trigrams does not have duplicates, if any trigram
/// belongs to more than one class it is only inserted once.
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
