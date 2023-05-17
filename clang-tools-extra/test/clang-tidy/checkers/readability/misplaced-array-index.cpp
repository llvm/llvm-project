// RUN: %check_clang_tidy %s readability-misplaced-array-index %t

#define ABC  "abc"

struct XY { int *X; int *Y; };

void dostuff(int);

void unusualSyntax(int *P1, struct XY *P2) {
  10[P1] = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: confusing array subscript expression, usually the index is inside the []
  // CHECK-FIXES: P1[10] = 0;

  10[P2->X] = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: confusing array subscript expression
  // CHECK-FIXES: P2->X[10] = 0;

  dostuff(1["abc"]);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: confusing array subscript expression
  // CHECK-FIXES:  dostuff("abc"[1]);

  dostuff(1[ABC]);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: confusing array subscript expression
  // CHECK-FIXES:  dostuff(ABC[1]);

  dostuff(0[0 + ABC]);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: confusing array subscript expression
  // CHECK-FIXES:  dostuff(0[0 + ABC]);
  // No fixit. Probably the code should be ABC[0]
}

void normalSyntax(int *X) {
  X[10] = 0;
}
