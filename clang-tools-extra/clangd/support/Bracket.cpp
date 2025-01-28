//===--- Bracket.cpp - Analyze bracket structure --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The basic phases of our bracket matching are:
//
// 1) A simple "greedy" match looks for well-nested subsequences.
//
//    We can't fully trust the results of this, consider:
//      while (1) {   // A
//        if (true) { // B
//          break;
//      }             // C
//    Greedy matching will match B=C, when we should at least consider A=C.
//    However for the correct parts of the file, the greedy match gives the
//    right answer. It produces useful candidates for phase 2.
//
//    simplePairBrackets handles this step.
//
// 2) Try to identify places where formatting indicates that the greedy match
//    was correct. This is similar to how a human would scan a large file.
//
//    For example:
//      int foo() {      // X
//        // indented
//        while (1) {
//          // valid code
//        }
//        return bar(42);
//      }                // Y
//    We can "verify" that X..Y looks like a braced block, and the greedy match
//    tells us that substring is perfectly nested.
//    We trust the pairings of those brackets and don't examine them further.
//    However in the first example above, we do not trust B=C because the brace
//    indentation is suspect.
//
//    FIXME: implement this step.
//
// 3) Run full best-match optimization on remaining brackets.
//
//    Conceptually, this considers all possible matchings and optimizes cost:
//      - there is a cost for failing to match a bracket
//      - there is a variable cost for matching two brackets.
//        (For example if brace indentation doesn't match).
//
//    In the first example we have three alternatives, and they are ranked:
//      1) A=C, skip B
//      2) B=C, skip A
//      3) skip A, skip B, skip C
//    The cost for skipping a bracket is high, so option 3 is worst.
//    B=C costs more than A=C, because the indentation doesn't match.
//
//    It would be correct to run this step alone, but it would be too slow.
//    The implementation is dynamic programming in N^3 space and N^2 time.
//    Having earlier steps filter out most brackets is key to performance.
//
//    FIXME: implement this step.
//
//===----------------------------------------------------------------------===//

#include "Bracket.h"

namespace clang {
namespace clangd {
namespace {

struct Bracket {
  using Index = unsigned;
  constexpr static Index None = -1;

  enum BracketKind : char { Paren, Brace, Square } Kind;
  enum Direction : bool { Open, Close } Dir;
  unsigned Line;
  unsigned Indent;
  Token::Index Tok;
  Bracket::Index Pair = None;
};

// Find brackets in the stream and convert to Bracket struct.
std::vector<Bracket> findBrackets(const TokenStream &Stream) {
  std::vector<Bracket> Brackets;
  auto Add = [&](const Token &Tok, Bracket::BracketKind K,
                 Bracket::Direction D) {
    Brackets.push_back(
        {K, D, Tok.Line, Tok.Indent, Stream.index(Tok), Bracket::None});
  };
  for (const auto &Tok : Stream.tokens()) {
    switch (Tok.Kind) {
    case clang::tok::l_paren:
      Add(Tok, Bracket::Paren, Bracket::Open);
      break;
    case clang::tok::r_paren:
      Add(Tok, Bracket::Paren, Bracket::Close);
      break;
    case clang::tok::l_brace:
      Add(Tok, Bracket::Brace, Bracket::Open);
      break;
    case clang::tok::r_brace:
      Add(Tok, Bracket::Brace, Bracket::Close);
      break;
    case clang::tok::l_square:
      Add(Tok, Bracket::Square, Bracket::Open);
      break;
    case clang::tok::r_square:
      Add(Tok, Bracket::Square, Bracket::Close);
      break;
    default:
      break;
    }
  }
  return Brackets;
}

// Write the bracket pairings from Brackets back to Tokens.
void applyPairings(ArrayRef<Bracket> Brackets, TokenStream &Tokens) {
  for (const auto &B : Brackets)
    Tokens.tokens()[B.Tok].Pair =
        (B.Pair == Bracket::None) ? 0 : (int32_t)Brackets[B.Pair].Tok - B.Tok;
}

// Find perfect pairings (ignoring whitespace) via greedy algorithm.
// This means two brackets are paired if they match and the brackets between
// them nest perfectly, with no skipped or crossed brackets.
void simplePairBrackets(MutableArrayRef<Bracket> Brackets) {
  std::vector<unsigned> Stack;
  for (unsigned I = 0; I < Brackets.size(); ++I) {
    if (Brackets[I].Dir == Bracket::Open) {
      Stack.push_back(I);
    } else if (!Stack.empty() &&
               Brackets[Stack.back()].Kind == Brackets[I].Kind) {
      Brackets[Stack.back()].Pair = I;
      Brackets[I].Pair = Stack.back();
      Stack.pop_back();
    } else {
      // Unpaired closer, no brackets on stack are part of a perfect sequence.
      Stack.clear();
    }
  }
  // Any remaining brackets on the stack stay unpaired.
}

} // namespace

void pairBrackets(TokenStream &Stream) {
  auto Brackets = findBrackets(Stream);
  simplePairBrackets(Brackets);
  applyPairings(Brackets, Stream);
}

} // namespace clangd
} // namespace clang
