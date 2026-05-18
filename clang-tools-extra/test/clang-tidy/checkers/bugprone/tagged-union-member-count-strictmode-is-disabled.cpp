// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: false, \
// RUN:  }}' --

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has more data members (2) than tags (1)
struct Incorrect {
  enum {
    tags1,
  } Tags;
  union {
    char A;
    short B;
  } Data;
};

struct CorrectBecauseStrictModeIsDisabled { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
  } Tags;
  union {
    char A;
    short B;
  } Data;
};
