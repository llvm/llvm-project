//===--- Trigram.cpp - Trigram generation for Fuzzy Matching ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Trigram.h"
#include "../../FuzzyMatch.h"
#include "Token.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"

#include <cctype>
#include <queue>
#include <string>

using namespace llvm;

namespace clang {
namespace clangd {
namespace dex {

// FIXME(kbobyrev): Deal with short symbol symbol names. A viable approach would
// be generating unigrams and bigrams here, too. This would prevent symbol index
// from applying fuzzy matching on a tremendous number of symbols and allow
// supplementary retrieval for short queries.
//
// Short names (total segment length <3 characters) are currently ignored.
std::vector<Token> generateIdentifierTrigrams(llvm::StringRef Identifier) {
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
  // * Skip-1-Next Head - front character of the skip-1-next segment
  //
  // Next stores tuples of three indices in the presented order, if a variant is
  // not available then 0 is stored.
  std::vector<std::array<unsigned, 3>> Next(LowercaseIdentifier.size());
  unsigned NextTail = 0, NextHead = 0, NextNextHead = 0;
  for (int I = LowercaseIdentifier.size() - 1; I >= 0; --I) {
    Next[I] = {{NextTail, NextHead, NextNextHead}};
    NextTail = Roles[I] == Tail ? I : 0;
    if (Roles[I] == Head) {
      NextNextHead = NextHead;
      NextHead = I;
    }
  }

  DenseSet<Token> UniqueTrigrams;
  std::array<char, 4> Chars;
  for (size_t I = 0; I < LowercaseIdentifier.size(); ++I) {
    // Skip delimiters.
    if (Roles[I] != Head && Roles[I] != Tail)
      continue;
    for (const unsigned J : Next[I]) {
      if (!J)
        continue;
      for (const unsigned K : Next[J]) {
        if (!K)
          continue;
        Chars = {{LowercaseIdentifier[I], LowercaseIdentifier[J],
                  LowercaseIdentifier[K], 0}};
        auto Trigram = Token(Token::Kind::Trigram, Chars.data());
        // Push unique trigrams to the result.
        if (!UniqueTrigrams.count(Trigram)) {
          UniqueTrigrams.insert(Trigram);
        }
      }
    }
  }

  std::vector<Token> Result;
  for (const auto &Trigram : UniqueTrigrams)
    Result.push_back(Trigram);

  return Result;
}

// FIXME(kbobyrev): Similarly, to generateIdentifierTrigrams, this ignores short
// inputs (total segment length <3 characters).
std::vector<Token> generateQueryTrigrams(llvm::StringRef Query) {
  // Apply fuzzy matching text segmentation.
  std::vector<CharRole> Roles(Query.size());
  calculateRoles(Query, llvm::makeMutableArrayRef(Roles.data(), Query.size()));

  std::string LowercaseQuery = Query.lower();

  DenseSet<Token> UniqueTrigrams;
  std::deque<char> Chars;

  for (size_t I = 0; I < LowercaseQuery.size(); ++I) {
    // If current symbol is delimiter, just skip it.
    if (Roles[I] != Head && Roles[I] != Tail)
      continue;

    Chars.push_back(LowercaseQuery[I]);

    if (Chars.size() > 3)
      Chars.pop_front();
    if (Chars.size() == 3) {
      auto Trigram =
          Token(Token::Kind::Trigram, std::string(begin(Chars), end(Chars)));
      // Push unique trigrams to the result.
      if (!UniqueTrigrams.count(Trigram)) {
        UniqueTrigrams.insert(Trigram);
      }
    }
  }

  std::vector<Token> Result;
  for (const auto &Trigram : UniqueTrigrams)
    Result.push_back(Trigram);

  return Result;
}

} // namespace dex
} // namespace clangd
} // namespace clang
