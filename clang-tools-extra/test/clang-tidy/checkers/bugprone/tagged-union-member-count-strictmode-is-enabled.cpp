// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-tagged-union-member-count %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-tagged-union-member-count.StrictMode: true, \
// RUN:  }}' --

// CHECK-MESSAGES: :[[@LINE+1]]:8: warning: tagged union has fewer data members (2) than tags (3)
struct IncorrectBecauseStrictmodeIsEnabled {
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

struct Correct { // No warnings expected
  enum {
    tags1,
    tags2,
    tags3,
  } Tags;
  union {
    char A;
    short B;
    int C;
  } Data;
};
