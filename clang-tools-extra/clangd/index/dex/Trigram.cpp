//===--- Trigram.cpp - Trigram generation for Fuzzy Matching ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Trigram.h"
#include "FuzzyMatch.h"
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

std::vector<Token> generateIdentifierTrigrams(StringRef Identifier) {
  // Apply fuzzy matching text segmentation.
  std::vector<CharRole> Roles(Identifier.size());
  calculateRoles(Identifier,
                 makeMutableArrayRef(Roles.data(), Identifier.size()));

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

  DenseSet<Token> UniqueTrigrams;

  auto Add = [&](std::string Chars) {
    UniqueTrigrams.insert(Token(Token::Kind::Trigram, Chars));
  };

  // Iterate through valid sequneces of three characters Fuzzy Matcher can
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
        Add({{LowercaseIdentifier[I], LowercaseIdentifier[J],
              LowercaseIdentifier[K]}});
      }
    }
  }
  // Emit short-query trigrams: FooBar -> f, fo, fb.
  if (!LowercaseIdentifier.empty())
    Add({LowercaseIdentifier[0]});
  if (LowercaseIdentifier.size() >= 2)
    Add({LowercaseIdentifier[0], LowercaseIdentifier[1]});
  for (size_t I = 1; I < LowercaseIdentifier.size(); ++I)
    if (Roles[I] == Head) {
      Add({LowercaseIdentifier[0], LowercaseIdentifier[I]});
      break;
    }

  return {UniqueTrigrams.begin(), UniqueTrigrams.end()};
}

std::vector<Token> generateQueryTrigrams(StringRef Query) {
  if (Query.empty())
    return {};
  std::string LowercaseQuery = Query.lower();
  if (Query.size() < 3) // short-query trigrams only
    return {Token(Token::Kind::Trigram, LowercaseQuery)};

  // Apply fuzzy matching text segmentation.
  std::vector<CharRole> Roles(Query.size());
  calculateRoles(Query, makeMutableArrayRef(Roles.data(), Query.size()));

  DenseSet<Token> UniqueTrigrams;
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
