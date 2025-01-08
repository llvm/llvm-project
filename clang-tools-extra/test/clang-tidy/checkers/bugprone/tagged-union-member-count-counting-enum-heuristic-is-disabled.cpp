// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:       bugprone-tagged-union-member-count.StrictMode: true, \
// RUN:       bugprone-tagged-union-member-count.EnableCountingEnumHeuristic: false, \
// RUN:   }}' --

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has fewer data members (3) than tags (4)
struct IncorrectBecauseHeuristicIsDisabledPrefixCase {
  enum {
    tags11,
    tags22,
    tags33,
    lasttag,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};

struct CorrectBecauseHeuristicIsDisabledPrefixCase { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
    lasttags,
  } Tags;
  union {
    char A;
    short B;
    int C;
    long D;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has fewer data members (3) than tags (4)
struct IncorrectBecauseHeuristicIsDisabledSuffixCase {
  enum {
    tags11,
    tags22,
    tags33,
    tags_count,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};

struct CorrectBecauseHeuristicIsDisabledSuffixCase { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
    tags_count,
  } Tags;
  union {
    char A;
    short B;
    int C;
    long D;
  } Data;
};
