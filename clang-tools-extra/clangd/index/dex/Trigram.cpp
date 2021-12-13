//===--- Trigram.cpp - Trigram generation for Fuzzy Matching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Trigram.h"
#include "FuzzyMatch.h"
#include "Token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include <cctype>
#include <limits>
#include <queue>
#include <string>
#include <utility>

namespace clang {
namespace clangd {
namespace dex {

// Produce trigrams (including duplicates) and pass them to Out().
template <typename Func>
static void identifierTrigrams(llvm::StringRef Identifier, Func Out) {
  assert(!Identifier.empty());
  // Apply fuzzy matching text segmentation.
  llvm::SmallVector<CharRole> Roles(Identifier.size());
  calculateRoles(Identifier,
                 llvm::makeMutableArrayRef(Roles.data(), Identifier.size()));

  std::string LowercaseIdentifier = Identifier.lower();

  // For each character, store indices of the characters to which fuzzy matching
  // algorithm can jump. There are 2 possible variants:
  //
  // * Next Tail - next character from the same segment
  // * Next Head - front character of the next segment
  //
  // Next stores tuples of three indices in the presented order, if a variant is
  // not available then 0 is stored.
  llvm::SmallVector<std::array<unsigned, 2>, 12> Next(
      LowercaseIdentifier.size());
  unsigned NextTail = 0, NextHead = 0;
  for (int I = LowercaseIdentifier.size() - 1; I >= 0; --I) {
    Next[I] = {{NextTail, NextHead}};
    NextTail = Roles[I] == Tail ? I : 0;
    if (Roles[I] == Head) {
      NextHead = I;
    }
  }

  // Iterate through valid sequences of three characters Fuzzy Matcher can
  // process.
  for (unsigned I = 0; I < LowercaseIdentifier.size(); ++I) {
    // Skip delimiters.
    if (Roles[I] != Head && Roles[I] != Tail)
      continue;
    for (unsigned J : Next[I]) {
      if (J == 0)
        continue;
      for (unsigned K : Next[J]) {
        if (K == 0)
          continue;
        Out(Trigram(LowercaseIdentifier[I], LowercaseIdentifier[J],
                    LowercaseIdentifier[K]));
      }
    }
  }
  // Short queries semantics are different. When the user dosn't type enough
  // symbols to form trigrams, we still want to serve meaningful results. To
  // achieve that, we form incomplete trigrams (bi- and unigrams) for the
  // identifiers and also generate short trigrams on the query side from what
  // is available. We allow a small number of short trigram types in order to
  // prevent excessive memory usage and increase the quality of the search.
  // Only the first few symbols are allowed to be used in incomplete trigrams.
  //
  // Example - for "_abc_def_ghi_jkl" we'll get following incomplete trigrams:
  // "_", "_a", "a", "ab", "ad", "d", "de", "dg"
  for (unsigned Position = 0, HeadsSeen = 0; HeadsSeen < 2;) {
    // The first symbol might be a separator, so the loop condition should be
    // stopping as soon as there is no next head or we have seen two heads.
    if (Roles[Position] == Head)
      ++HeadsSeen;
    Out(Trigram(LowercaseIdentifier[Position]));
    for (unsigned I : Next[Position])
      if (I != 0)
        Out(Trigram(LowercaseIdentifier[Position], LowercaseIdentifier[I]));
    Position = Next[Position][1];
    if (Position == 0)
      break;
  }
}

void generateIdentifierTrigrams(llvm::StringRef Identifier,
                                std::vector<Trigram> &Result) {
  // Empirically, scanning for duplicates is faster with fewer trigrams, and
  // a hashtable is faster with more. This is a hot path, so dispatch based on
  // expected number of trigrams. Longer identifiers produce more trigrams.
  // The magic number was tuned by running IndexBenchmark.DexBuild.
  constexpr unsigned ManyTrigramsIdentifierThreshold = 14;
  Result.clear();
  if (Identifier.empty())
    return;

  if (Identifier.size() < ManyTrigramsIdentifierThreshold) {
    identifierTrigrams(Identifier, [&](Trigram T) {
      if (!llvm::is_contained(Result, T))
        Result.push_back(T);
    });
  } else {
    identifierTrigrams(Identifier, [&](Trigram T) { Result.push_back(T); });
    llvm::sort(Result);
    Result.erase(std::unique(Result.begin(), Result.end()), Result.end());
  }
}

std::vector<Token> generateQueryTrigrams(llvm::StringRef Query) {
  if (Query.empty())
    return {};

  // Apply fuzzy matching text segmentation.
  llvm::SmallVector<CharRole> Roles(Query.size());
  calculateRoles(Query, llvm::makeMutableArrayRef(Roles.data(), Query.size()));

  std::string LowercaseQuery = Query.lower();

  llvm::DenseSet<Token> UniqueTrigrams;
  std::string Chars;
  for (unsigned I = 0; I < LowercaseQuery.size(); ++I) {
    if (Roles[I] != Head && Roles[I] != Tail)
      continue; // Skip delimiters.
    Chars.push_back(LowercaseQuery[I]);
    if (Chars.size() > 3)
      Chars.erase(Chars.begin());
    if (Chars.size() == 3)
      UniqueTrigrams.insert(Token(Token::Kind::Trigram, Chars));
  }

  // For queries with very few letters, generateIdentifierTrigrams emulates
  // outputs of this function to match the semantics.
  if (UniqueTrigrams.empty()) {
    // If bigram can't be formed out of letters/numbers, we prepend separator.
    std::string Result(1, LowercaseQuery.front());
    for (unsigned I = 1; I < LowercaseQuery.size(); ++I)
      if (Roles[I] == Head || Roles[I] == Tail)
        Result += LowercaseQuery[I];
    UniqueTrigrams.insert(
        Token(Token::Kind::Trigram, llvm::StringRef(Result).take_back(2)));
  }

  return {UniqueTrigrams.begin(), UniqueTrigrams.end()};
}

} // namespace dex
} // namespace clangd
} // namespace clang
