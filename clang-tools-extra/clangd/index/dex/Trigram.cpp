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

/// This is used to mark unigrams and bigrams and distinct them from complete
/// trigrams. Since '$' is not present in valid identifier names, it is safe to
/// use it as the special symbol.
static const char END_MARKER = '$';

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
  // Store two first HEAD characters in the identifier (if present).
  std::deque<char> TwoHeads;
  for (int I = LowercaseIdentifier.size() - 1; I >= 0; --I) {
    Next[I] = {{NextTail, NextHead, NextNextHead}};
    NextTail = Roles[I] == Tail ? I : 0;
    if (Roles[I] == Head) {
      NextNextHead = NextHead;
      NextHead = I;
      TwoHeads.push_front(LowercaseIdentifier[I]);
      if (TwoHeads.size() > 2)
        TwoHeads.pop_back();
    }
  }

  DenseSet<Token> UniqueTrigrams;

  auto add = [&](std::string Chars) {
    UniqueTrigrams.insert(Token(Token::Kind::Trigram, Chars));
  };

  if (TwoHeads.size() == 2)
    add({{TwoHeads.front(), TwoHeads.back(), END_MARKER}});

  if (!LowercaseIdentifier.empty())
    add({{LowercaseIdentifier.front(), END_MARKER, END_MARKER}});

  if (LowercaseIdentifier.size() >= 2)
    add({{LowercaseIdentifier[0], LowercaseIdentifier[1], END_MARKER}});

  if (LowercaseIdentifier.size() >= 3)
    add({{LowercaseIdentifier[0], LowercaseIdentifier[1],
          LowercaseIdentifier[2]}});

  // Iterate through valid seqneces of three characters Fuzzy Matcher can
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
        add({{LowercaseIdentifier[I], LowercaseIdentifier[J],
              LowercaseIdentifier[K]}});
      }
    }
  }

  std::vector<Token> Result;
  for (const auto &Trigram : UniqueTrigrams)
    Result.push_back(Trigram);

  return Result;
}

std::vector<Token> generateQueryTrigrams(llvm::StringRef Query) {
  // Apply fuzzy matching text segmentation.
  std::vector<CharRole> Roles(Query.size());
  calculateRoles(Query, llvm::makeMutableArrayRef(Roles.data(), Query.size()));

  // Additional pass is necessary to count valid identifier characters.
  // Depending on that, this function might return incomplete trigram.
  unsigned ValidSymbolsCount = 0;
  for (const auto Role : Roles)
    if (Role == Head || Role == Tail)
      ++ValidSymbolsCount;

  std::string LowercaseQuery = Query.lower();

  DenseSet<Token> UniqueTrigrams;

  // If the number of symbols which can form fuzzy matching trigram is not
  // sufficient, generate a single incomplete trigram for query.
  if (ValidSymbolsCount < 3) {
    std::string Chars =
        LowercaseQuery.substr(0, std::min<size_t>(3UL, Query.size()));
    Chars.append(3 - Chars.size(), END_MARKER);
    UniqueTrigrams.insert(Token(Token::Kind::Trigram, Chars));
  } else {
    std::deque<char> Chars;
    for (size_t I = 0; I < LowercaseQuery.size(); ++I) {
      // If current symbol is delimiter, just skip it.
      if (Roles[I] != Head && Roles[I] != Tail)
        continue;

      Chars.push_back(LowercaseQuery[I]);

      if (Chars.size() > 3)
        Chars.pop_front();

      if (Chars.size() == 3) {
        UniqueTrigrams.insert(
            Token(Token::Kind::Trigram, std::string(begin(Chars), end(Chars))));
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
