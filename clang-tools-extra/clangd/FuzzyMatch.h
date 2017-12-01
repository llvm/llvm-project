//===--- FuzzyMatch.h - Approximate identifier matching  ---------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements fuzzy-matching of strings against identifiers.
// It indicates both the existence and quality of a match:
// 'eb' matches both 'emplace_back' and 'embed', the former has a better score.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FUZZYMATCH_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FUZZYMATCH_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

// A matcher capable of matching and scoring strings against a single pattern.
// It's optimized for matching against many strings - match() does not allocate.
class FuzzyMatcher {
public:
  // Characters beyond MaxPat are ignored.
  FuzzyMatcher(llvm::StringRef Pattern);

  // If Word matches the pattern, return a score in [0,1] (higher is better).
  // Characters beyond MaxWord are ignored.
  llvm::Optional<float> match(llvm::StringRef Word);

  // Dump internal state from the last match() to the stream, for debugging.
  // Returns the pattern with [] around matched characters, e.g.
  //   [u_p] + "unique_ptr" --> "[u]nique[_p]tr"
  llvm::SmallString<256> dumpLast(llvm::raw_ostream &) const;

private:
  // We truncate the pattern and the word to bound the cost of matching.
  constexpr static int MaxPat = 63, MaxWord = 127;
  enum CharRole : char; // For segmentation.
  enum CharType : char; // For segmentation.
  enum Action { Miss = 0, Match = 1 };

  bool init(llvm::StringRef Word);
  void buildGraph();
  void calculateRoles(const char *Text, CharRole *Out, int N);
  int skipPenalty(int W, Action Last);
  int matchBonus(int P, int W, Action Last);

  // Pattern data is initialized by the constructor, then constant.
  char Pat[MaxPat];         // Pattern data
  int PatN;                 // Length
  char LowPat[MaxPat];      // Pattern in lowercase
  CharRole PatRole[MaxPat]; // Pattern segmentation info
  bool CaseSensitive;       // Case-sensitive match if pattern has uppercase
  float ScoreScale;         // Normalizes scores for the pattern length.

  // Word data is initialized on each call to match(), mostly by init().
  char Word[MaxWord];         // Word data
  int WordN;                  // Length
  char LowWord[MaxWord];      // Word in lowercase
  CharRole WordRole[MaxWord]; // Word segmentation info
  bool WordContainsPattern;   // Simple substring check

  // Cumulative best-match score table.
  // Boundary conditions are filled in by the constructor.
  // The rest is repopulated for each match(), by buildGraph().
  struct ScoreInfo {
    signed int Score : 15;
    Action Prev : 1;
  };
  ScoreInfo Scores[MaxPat + 1][MaxWord + 1][/* Last Action */ 2];
};

} // namespace clangd
} // namespace clang

#endif
