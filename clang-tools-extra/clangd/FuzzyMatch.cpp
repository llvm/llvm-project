//===--- FuzzyMatch.h - Approximate identifier matching  ---------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// To check for a match between a Pattern ('u_p') and a Word ('unique_ptr'),
// we consider the possible partial match states:
//
//     u n i q u e _ p t r
//   +---------------------
//   |A . . . . . . . . . .
//  u|
//   |. . . . . . . . . . .
//  _|
//   |. . . . . . . O . . .
//  p|
//   |. . . . . . . . . . B
//
// Each dot represents some prefix of the pattern being matched against some
// prefix of the word.
//   - A is the initial state: '' matched against ''
//   - O is an intermediate state: 'u_' matched against 'unique_'
//   - B is the target state: 'u_p' matched against 'unique_ptr'
//
// We aim to find the best path from A->B.
//  - Moving right (consuming a word character)
//    Always legal: not all word characters must match.
//  - Moving diagonally (consuming both a word and pattern character)
//    Legal if the characters match.
//  - Moving down (consuming a pattern character) is never legal.
//    Never legal: all pattern characters must match something.
// Characters are matched case-insensitively.
// The first pattern character may only match the start of a word segment.
//
// The scoring is based on heuristics:
//  - when matching a character, apply a bonus or penalty depending on the
//    match quality (does case match, do word segments align, etc)
//  - when skipping a character, apply a penalty if it hurts the match
//    (it starts a word segment, or splits the matched region, etc)
//
// These heuristics require the ability to "look backward" one character, to
// see whether it was matched or not. Therefore the dynamic-programming matrix
// has an extra dimension (last character matched).
// Each entry also has an additional flag indicating whether the last-but-one
// character matched, which is needed to trace back through the scoring table
// and reconstruct the match.
//
// We treat strings as byte-sequences, so only ASCII has first-class support.
//
// This algorithm was inspired by VS code's client-side filtering, and aims
// to be mostly-compatible.
//
//===----------------------------------------------------------------------===//

#include "FuzzyMatch.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Format.h"

namespace clang {
namespace clangd {
using namespace llvm;

constexpr int FuzzyMatcher::MaxPat;
constexpr int FuzzyMatcher::MaxWord;

static char lower(char C) { return C >= 'A' && C <= 'Z' ? C + ('a' - 'A') : C; }
// A "negative infinity" score that won't overflow.
// We use this to mark unreachable states and forbidden solutions.
// Score field is 15 bits wide, min value is -2^14, we use half of that.
static constexpr int AwfulScore = -(1 << 13);
static bool isAwful(int S) { return S < AwfulScore / 2; }
static constexpr int PerfectBonus = 3; // Perfect per-pattern-char score.

FuzzyMatcher::FuzzyMatcher(StringRef Pattern)
    : PatN(std::min<int>(MaxPat, Pattern.size())),
      ScoreScale(PatN ? float{1} / (PerfectBonus * PatN) : 0), WordN(0) {
  std::copy(Pattern.begin(), Pattern.begin() + PatN, Pat);
  for (int I = 0; I < PatN; ++I)
    LowPat[I] = lower(Pat[I]);
  Scores[0][0][Miss] = {0, Miss};
  Scores[0][0][Match] = {AwfulScore, Miss};
  for (int P = 0; P <= PatN; ++P)
    for (int W = 0; W < P; ++W)
      for (Action A : {Miss, Match})
        Scores[P][W][A] = {AwfulScore, Miss};
  if (PatN > 0)
    calculateRoles(Pat, PatRole, PatTypeSet, PatN);
}

Optional<float> FuzzyMatcher::match(StringRef Word) {
  if (!(WordContainsPattern = init(Word)))
    return None;
  if (!PatN)
    return 1;
  buildGraph();
  auto Best = std::max(Scores[PatN][WordN][Miss].Score,
                       Scores[PatN][WordN][Match].Score);
  if (isAwful(Best))
    return None;
  float Score =
      ScoreScale * std::min(PerfectBonus * PatN, std::max<int>(0, Best));
  // If the pattern is as long as the word, we have an exact string match,
  // since every pattern character must match something.
  if (WordN == PatN)
    Score *= 2; // May not be perfect 2 if case differs in a significant way.
  return Score;
}

// Segmentation of words and patterns.
// A name like "fooBar_baz" consists of several parts foo, bar, baz.
// Aligning segmentation of word and pattern improves the fuzzy-match.
// For example: [lol] matches "LaughingOutLoud" better than "LionPopulation"
//
// First we classify each character into types (uppercase, lowercase, etc).
// Then we look at the sequence: e.g. [upper, lower] is the start of a segment.

// We only distinguish the types of characters that affect segmentation.
// It's not obvious how to segment digits, we treat them as lowercase letters.
// As we don't decode UTF-8, we treat bytes over 127 as lowercase too.
// This means we require exact (case-sensitive) match.
enum FuzzyMatcher::CharType : unsigned char {
  Empty = 0,       // Before-the-start and after-the-end (and control chars).
  Lower = 1,       // Lowercase letters, digits, and non-ASCII bytes.
  Upper = 2,       // Uppercase letters.
  Punctuation = 3, // ASCII punctuation (including Space)
};

// We get CharTypes from a lookup table. Each is 2 bits, 4 fit in each byte.
// The top 6 bits of the char select the byte, the bottom 2 select the offset.
// e.g. 'q' = 010100 01 = byte 28 (55), bits 3-2 (01) -> Lower.
constexpr static uint8_t CharTypes[] = {
    0x00, 0x00, 0x00, 0x00, // Control characters
    0x00, 0x00, 0x00, 0x00, // Control characters
    0xff, 0xff, 0xff, 0xff, // Punctuation
    0x55, 0x55, 0xf5, 0xff, // Numbers->Lower, more Punctuation.
    0xab, 0xaa, 0xaa, 0xaa, // @ and A-O
    0xaa, 0xaa, 0xea, 0xff, // P-Z, more Punctuation.
    0x57, 0x55, 0x55, 0x55, // ` and a-o
    0x55, 0x55, 0xd5, 0x3f, // p-z, Punctuation, DEL.
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, // Bytes over 127 -> Lower.
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, // (probably UTF-8).
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
};

// Each character's Role is the Head or Tail of a segment, or a Separator.
// e.g. XMLHttpRequest_Async
//      +--+---+------ +----
//      ^Head   ^Tail ^Separator
enum FuzzyMatcher::CharRole : unsigned char {
  Unknown = 0,   // Stray control characters or impossible states.
  Tail = 1,      // Part of a word segment, but not the first character.
  Head = 2,      // The first character of a word segment.
  Separator = 3, // Punctuation characters that separate word segments.
};

// The Role can be determined from the Type of a character and its neighbors:
//
//   Example  | Chars | Type | Role
//   ---------+--------------+-----
//   F(o)oBar | Foo   | Ull  | Tail
//   Foo(B)ar | oBa   | lUl  | Head
//   (f)oo    | ^fo   | Ell  | Head
//   H(T)TP   | HTT   | UUU  | Tail
//
// Our lookup table maps a 6 bit key (Prev, Curr, Next) to a 2-bit Role.
// A byte packs 4 Roles. (Prev, Curr) selects a byte, Next selects the offset.
// e.g. Lower, Upper, Lower -> 01 10 01 -> byte 6 (aa), bits 3-2 (10) -> Head.
constexpr static uint8_t CharRoles[] = {
    // clang-format off
    //         Curr= Empty Lower Upper Separ
    /* Prev=Empty */ 0x00, 0xaa, 0xaa, 0xff, // At start, Lower|Upper->Head
    /* Prev=Lower */ 0x00, 0x55, 0xaa, 0xff, // In word, Upper->Head;Lower->Tail
    /* Prev=Upper */ 0x00, 0x55, 0x59, 0xff, // Ditto, but U(U)U->Tail
    /* Prev=Separ */ 0x00, 0xaa, 0xaa, 0xff, // After separator, like at start
    // clang-format on
};

template <typename T> static T packedLookup(const uint8_t *Data, int I) {
  return static_cast<T>((Data[I >> 2] >> ((I & 3) * 2)) & 3);
}
void FuzzyMatcher::calculateRoles(const char *Text, CharRole *Out, int &TypeSet,
                                  int N) {
  assert(N > 0);
  CharType Type = packedLookup<CharType>(CharTypes, Text[0]);
  TypeSet = 1 << Type;
  // Types holds a sliding window of (Prev, Curr, Next) types.
  // Initial value is (Empty, Empty, type of Text[0]).
  int Types = Type;
  // Rotate slides in the type of the next character.
  auto Rotate = [&](CharType T) { Types = ((Types << 2) | T) & 0x3f; };
  for (int I = 0; I < N - 1; ++I) {
    // For each character, rotate in the next, and look up the role.
    Type = packedLookup<CharType>(CharTypes, Text[I + 1]);
    TypeSet |= 1 << Type;
    Rotate(Type);
    *Out++ = packedLookup<CharRole>(CharRoles, Types);
  }
  // For the last character, the "next character" is Empty.
  Rotate(Empty);
  *Out++ = packedLookup<CharRole>(CharRoles, Types);
}

// Sets up the data structures matching Word.
// Returns false if we can cheaply determine that no match is possible.
bool FuzzyMatcher::init(StringRef NewWord) {
  WordN = std::min<int>(MaxWord, NewWord.size());
  if (PatN > WordN)
    return false;
  std::copy(NewWord.begin(), NewWord.begin() + WordN, Word);
  if (PatN == 0)
    return true;
  for (int I = 0; I < WordN; ++I)
    LowWord[I] = lower(Word[I]);

  // Cheap subsequence check.
  for (int W = 0, P = 0; P != PatN; ++W) {
    if (W == WordN)
      return false;
    if (LowWord[W] == LowPat[P])
      ++P;
  }

  // FIXME: some words are hard to tokenize algorithmically.
  // e.g. vsprintf is V S Print F, and should match [pri] but not [int].
  // We could add a tokenization dictionary for common stdlib names.
  calculateRoles(Word, WordRole, WordTypeSet, WordN);
  return true;
}

// The forwards pass finds the mappings of Pattern onto Word.
// Score = best score achieved matching Word[..W] against Pat[..P].
// Unlike other tables, indices range from 0 to N *inclusive*
// Matched = whether we chose to match Word[W] with Pat[P] or not.
//
// Points are mostly assigned to matched characters, with 1 being a good score
// and 3 being a great one. So we treat the score range as [0, 3 * PatN].
// This range is not strict: we can apply larger bonuses/penalties, or penalize
// non-matched characters.
void FuzzyMatcher::buildGraph() {
  for (int W = 0; W < WordN; ++W) {
    Scores[0][W + 1][Miss] = {Scores[0][W][Miss].Score - skipPenalty(W, Miss),
                              Miss};
    Scores[0][W + 1][Match] = {AwfulScore, Miss};
  }
  for (int P = 0; P < PatN; ++P) {
    for (int W = P; W < WordN; ++W) {
      auto &Score = Scores[P + 1][W + 1], &PreMiss = Scores[P + 1][W];

      auto MatchMissScore = PreMiss[Match].Score;
      auto MissMissScore = PreMiss[Miss].Score;
      if (P < PatN - 1) { // Skipping trailing characters is always free.
        MatchMissScore -= skipPenalty(W, Match);
        MissMissScore -= skipPenalty(W, Miss);
      }
      Score[Miss] = (MatchMissScore > MissMissScore)
                        ? ScoreInfo{MatchMissScore, Match}
                        : ScoreInfo{MissMissScore, Miss};

      auto &PreMatch = Scores[P][W];
      auto MatchMatchScore =
          allowMatch(P, W, Match)
              ? PreMatch[Match].Score + matchBonus(P, W, Match)
              : AwfulScore;
      auto MissMatchScore = allowMatch(P, W, Miss)
                                ? PreMatch[Miss].Score + matchBonus(P, W, Miss)
                                : AwfulScore;
      Score[Match] = (MatchMatchScore > MissMatchScore)
                         ? ScoreInfo{MatchMatchScore, Match}
                         : ScoreInfo{MissMatchScore, Miss};
    }
  }
}

bool FuzzyMatcher::allowMatch(int P, int W, Action Last) const {
  if (LowPat[P] != LowWord[W])
    return false;
  // We require a "strong" match:
  // - for the first pattern character.  [foo] !~ "barefoot"
  // - after a gap.                      [pat] !~ "patnther"
  if (Last == Miss) {
    // We're banning matches outright, so conservatively accept some other cases
    // where our segmentation might be wrong:
    //  - allow matching B in ABCDef (but not in NDEBUG)
    //  - we'd like to accept print in sprintf, but too many false positives
    if (WordRole[W] == Tail &&
        (Word[W] == LowWord[W] || !(WordTypeSet & 1 << Lower)))
      return false;
  }
  return true;
}

int FuzzyMatcher::skipPenalty(int W, Action Last) const {
  int S = 0;
  if (WordRole[W] == Head) // Skipping a segment.
    S += 1;
  if (Last == Match) // Non-consecutive match.
    S += 2;          // We'd rather skip a segment than split our match.
  return S;
}

int FuzzyMatcher::matchBonus(int P, int W, Action Last) const {
  assert(LowPat[P] == LowWord[W]);
  int S = 1;
  // Bonus: pattern so far is a (case-insensitive) prefix of the word.
  if (P == W) // We can't skip pattern characters, so we must have matched all.
    ++S;
  // Bonus: case matches, or a Head in the pattern aligns with one in the word.
  if ((Pat[P] == Word[W] && ((PatTypeSet & 1 << Upper) || P == W)) ||
      (PatRole[P] == Head && WordRole[W] == Head))
    ++S;
  // Penalty: matching inside a segment (and previous char wasn't matched).
  if (WordRole[W] == Tail && P && Last == Miss)
    S -= 3;
  // Penalty: a Head in the pattern matches in the middle of a word segment.
  if (PatRole[P] == Head && WordRole[W] == Tail)
    --S;
  // Penalty: matching the first pattern character in the middle of a segment.
  if (P == 0 && WordRole[W] == Tail)
    S -= 4;
  assert(S <= PerfectBonus);
  return S;
}

llvm::SmallString<256> FuzzyMatcher::dumpLast(llvm::raw_ostream &OS) const {
  llvm::SmallString<256> Result;
  OS << "=== Match \"" << StringRef(Word, WordN) << "\" against ["
     << StringRef(Pat, PatN) << "] ===\n";
  if (PatN == 0) {
    OS << "Pattern is empty: perfect match.\n";
    return Result = StringRef(Word, WordN);
  }
  if (WordN == 0) {
    OS << "Word is empty: no match.\n";
    return Result;
  }
  if (!WordContainsPattern) {
    OS << "Substring check failed.\n";
    return Result;
  } else if (isAwful(std::max(Scores[PatN][WordN][Match].Score,
                              Scores[PatN][WordN][Miss].Score))) {
    OS << "Substring check passed, but all matches are forbidden\n";
  }
  if (!(PatTypeSet & 1 << Upper))
    OS << "Lowercase query, so scoring ignores case\n";

  // Traverse Matched table backwards to reconstruct the Pattern/Word mapping.
  // The Score table has cumulative scores, subtracting along this path gives
  // us the per-letter scores.
  Action Last =
      (Scores[PatN][WordN][Match].Score > Scores[PatN][WordN][Miss].Score)
          ? Match
          : Miss;
  int S[MaxWord];
  Action A[MaxWord];
  for (int W = WordN - 1, P = PatN - 1; W >= 0; --W) {
    A[W] = Last;
    const auto &Cell = Scores[P + 1][W + 1][Last];
    if (Last == Match)
      --P;
    const auto &Prev = Scores[P + 1][W][Cell.Prev];
    S[W] = Cell.Score - Prev.Score;
    Last = Cell.Prev;
  }
  for (int I = 0; I < WordN; ++I) {
    if (A[I] == Match && (I == 0 || A[I - 1] == Miss))
      Result.push_back('[');
    if (A[I] == Miss && I > 0 && A[I - 1] == Match)
      Result.push_back(']');
    Result.push_back(Word[I]);
  }
  if (A[WordN - 1] == Match)
    Result.push_back(']');

  for (char C : StringRef(Word, WordN))
    OS << " " << C << " ";
  OS << "\n";
  for (int I = 0, J = 0; I < WordN; I++)
    OS << " " << (A[I] == Match ? Pat[J++] : ' ') << " ";
  OS << "\n";
  for (int I = 0; I < WordN; I++)
    OS << format("%2d ", S[I]);
  OS << "\n";

  OS << "\nSegmentation:";
  OS << "\n'" << StringRef(Word, WordN) << "'\n ";
  for (int I = 0; I < WordN; ++I)
    OS << "?-+ "[static_cast<int>(WordRole[I])];
  OS << "\n[" << StringRef(Pat, PatN) << "]\n ";
  for (int I = 0; I < PatN; ++I)
    OS << "?-+ "[static_cast<int>(PatRole[I])];
  OS << "\n";

  OS << "\nScoring table (last-Miss, last-Match):\n";
  OS << " |    ";
  for (char C : StringRef(Word, WordN))
    OS << "  " << C << " ";
  OS << "\n";
  OS << "-+----" << std::string(WordN * 4, '-') << "\n";
  for (int I = 0; I <= PatN; ++I) {
    for (Action A : {Miss, Match}) {
      OS << ((I && A == Miss) ? Pat[I - 1] : ' ') << "|";
      for (int J = 0; J <= WordN; ++J) {
        if (!isAwful(Scores[I][J][A].Score))
          OS << format("%3d%c", Scores[I][J][A].Score,
                       Scores[I][J][A].Prev == Match ? '*' : ' ');
        else
          OS << "    ";
      }
      OS << "\n";
    }
  }

  return Result;
}

} // namespace clangd
} // namespace clang
