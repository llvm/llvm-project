//===--- FuzzyMatch.h - Approximate identifier matching  ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

// Utilities for word segmentation.
// FuzzyMatcher already incorporates this logic, so most users don't need this.
//
// A name like "fooBar_baz" consists of several parts foo, bar, baz.
// Aligning segmentation of word and pattern improves the fuzzy-match.
// For example: [lol] matches "LaughingOutLoud" better than "LionPopulation"
//
// First we classify each character into types (uppercase, lowercase, etc).
// Then we look at the sequence: e.g. [upper, lower] is the start of a segment.

// We distinguish the types of characters that affect segmentation.
// It's not obvious how to segment digits, we treat them as lowercase letters.
// As we don't decode UTF-8, we treat bytes over 127 as lowercase too.
// This means we require exact (case-sensitive) match for those characters.
enum CharType : unsigned char {
  Empty = 0,       // Before-the-start and after-the-end (and control chars).
  Lower = 1,       // Lowercase letters, digits, and non-ASCII bytes.
  Upper = 2,       // Uppercase letters.
  Punctuation = 3, // ASCII punctuation (including Space)
};
// A CharTypeSet is a bitfield representing all the character types in a word.
// Its bits are 1<<Empty, 1<<Lower, etc.
using CharTypeSet = unsigned char;

// Each character's Role is the Head or Tail of a segment, or a Separator.
// e.g. XMLHttpRequest_Async
//      +--+---+------ +----
//      ^Head   ^Tail ^Separator
enum CharRole : unsigned char {
  Unknown = 0,   // Stray control characters or impossible states.
  Tail = 1,      // Part of a word segment, but not the first character.
  Head = 2,      // The first character of a word segment.
  Separator = 3, // Punctuation characters that separate word segments.
};

// Compute segmentation of Text.
// Character roles are stored in Roles (Roles.size() must equal Text.size()).
// The set of character types encountered is returned, this may inform
// heuristics for dealing with poorly-segmented identifiers like "strndup".
CharTypeSet calculateRoles(llvm::StringRef Text,
                           llvm::MutableArrayRef<CharRole> Roles);

// A matcher capable of matching and scoring strings against a single pattern.
// It's optimized for matching against many strings - match() does not allocate.
class FuzzyMatcher {
public:
  // Characters beyond MaxPat are ignored.
  FuzzyMatcher(llvm::StringRef Pattern);

  // If Word matches the pattern, return a score indicating the quality match.
  // Scores usually fall in a [0,1] range, with 1 being a very good score.
  // "Super" scores in (1,2] are possible if the pattern is the full word.
  // Characters beyond MaxWord are ignored.
  llvm::Optional<float> match(llvm::StringRef Word);

  llvm::StringRef pattern() const { return llvm::StringRef(Pat, PatN); }
  bool empty() const { return PatN == 0; }

  // Dump internal state from the last match() to the stream, for debugging.
  // Returns the pattern with [] around matched characters, e.g.
  //   [u_p] + "unique_ptr" --> "[u]nique[_p]tr"
  llvm::SmallString<256> dumpLast(llvm::raw_ostream &) const;

private:
  // We truncate the pattern and the word to bound the cost of matching.
  constexpr static int MaxPat = 63, MaxWord = 127;
  // Action describes how a word character was matched to the pattern.
  // It should be an enum, but this causes bitfield problems:
  //   - for MSVC the enum type must be explicitly unsigned for correctness
  //   - GCC 4.8 complains not all values fit if the type is unsigned
  using Action = bool;
  constexpr static Action Miss = false; // Word character was skipped.
  constexpr static Action Match = true; // Matched against a pattern character.

  bool init(llvm::StringRef Word);
  void buildGraph();
  bool allowMatch(int P, int W, Action Last) const;
  int skipPenalty(int W, Action Last) const;
  int matchBonus(int P, int W, Action Last) const;

  // Pattern data is initialized by the constructor, then constant.
  char Pat[MaxPat];         // Pattern data
  int PatN;                 // Length
  char LowPat[MaxPat];      // Pattern in lowercase
  CharRole PatRole[MaxPat]; // Pattern segmentation info
  CharTypeSet PatTypeSet;   // Bitmask of 1<<CharType for all Pattern characters
  float ScoreScale;         // Normalizes scores for the pattern length.

  // Word data is initialized on each call to match(), mostly by init().
  char Word[MaxWord];         // Word data
  int WordN;                  // Length
  char LowWord[MaxWord];      // Word in lowercase
  CharRole WordRole[MaxWord]; // Word segmentation info
  CharTypeSet WordTypeSet;    // Bitmask of 1<<CharType for all Word characters
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
