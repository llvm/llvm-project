// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictModeIsEnabled: 0, \
// RUN:     bugprone-tagged-union-member-count.CountingEnumHeuristicIsEnabled: 1, \
// RUN:     bugprone-tagged-union-member-count.CountingEnumSuffixes: "count", \
// RUN:     bugprone-tagged-union-member-count.CountingEnumPrefixes: "last", \
// RUN:  }}' --

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (3) than tags (2)
struct IncorrectBecauseHeuristicIsEnabledPrefixCase {
  enum {
    tags1,
    tags2,
    lasttag,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};

struct CorrectBecauseHeuristicIsEnabledPrefixCase { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
    lasttag,
  } Tags;
  union {
    int A;
    int B;
    int C;
  } Data;
};

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: Tagged union has more data members (3) than tags (2)
struct IncorrectBecauseHeuristicIsEnabledSuffixCase {
  enum {
    tags1,
    tags2,
    tags_count,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};

struct CorrectBecauseHeuristicIsEnabledSuffixCase { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
    tags_count,
  } Tags;
  union {
    int A;
    int B;
    int C;
  } Data;
};
