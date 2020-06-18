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
#include "llvm/ADT/StringExtras.h"
#include <cctype>
#include <queue>
#include <string>

namespace clang {
namespace clangd {
namespace dex {

// Produce trigrams (including duplicates) and pass them to Out().
template <typename Func>
static void identifierTrigrams(llvm::StringRef Identifier, Func Out) {
  // Apply fuzzy matching text segmentation.
  std::vector<CharRole> Roles(Identifier.size());
  calculateRoles(Identifier,
                 llvm::makeMutableArrayRef(Roles.data(), Identifier.size()));

  std::string LowercaseIdentifier = Identifier.lower();

  // For each character, store indices of the characters to which fuzzy matching
  // algorithm can jump. There are 3 possible variants:
  //
  // * Next Tail - next character from the same segment
  // * Next Head - front character of the next segment
  //
  // Next stores tuples of three indices in the presented order, if a variant is
  // not available then 0 is stored.
  std::vector<std::array<unsigned, 3>> Next(LowercaseIdentifier.size());
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
  for (size_t I = 0; I < LowercaseIdentifier.size(); ++I) {
    // Skip delimiters.
    if (Roles[I] != Head && Roles[I] != Tail)
      continue;
    for (const unsigned J : Next[I]) {
      if (J == 0)
        continue;
      for (const unsigned K : Next[J]) {
        if (K == 0)
          continue;
        Out(Trigram(LowercaseIdentifier[I], LowercaseIdentifier[J],
                    LowercaseIdentifier[K]));
      }
    }
  }
  // Emit short-query trigrams: FooBar -> f, fo, fb.
  if (!LowercaseIdentifier.empty())
    Out(Trigram(LowercaseIdentifier[0]));
  if (LowercaseIdentifier.size() >= 2)
    Out(Trigram(LowercaseIdentifier[0], LowercaseIdentifier[1]));
  for (size_t I = 1; I < LowercaseIdentifier.size(); ++I)
    if (Roles[I] == Head) {
      Out(Trigram(LowercaseIdentifier[0], LowercaseIdentifier[I]));
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
  std::string LowercaseQuery = Query.lower();
  if (Query.size() < 3) // short-query trigrams only
    return {Token(Token::Kind::Trigram, LowercaseQuery)};

  // Apply fuzzy matching text segmentation.
  std::vector<CharRole> Roles(Query.size());
  calculateRoles(Query, llvm::makeMutableArrayRef(Roles.data(), Query.size()));

  llvm::DenseSet<Token> UniqueTrigrams;
  std::string Chars;
  for (unsigned I = 0; I < Query.size(); ++I) {
    if (Roles[I] != Head && Roles[I] != Tail)
      continue; // Skip delimiters.
    Chars.push_back(LowercaseQuery[I]);
    if (Chars.size() > 3)
      Chars.erase(Chars.begin());
    if (Chars.size() == 3)
      UniqueTrigrams.insert(Token(Token::Kind::Trigram, Chars));
  }

  return {UniqueTrigrams.begin(), UniqueTrigrams.end()};
}

} // namespace dex
} // namespace clangd
} // namespace clang
