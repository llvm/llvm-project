// RUN: %check_clang_tidy %s misc-redundant-expression %t

struct Point {
  int x;
  int y;
  int a[5];
} P;

extern Point P1;
extern Point P2;

extern int foo(int x);
extern int bar(int x);
extern int bat(int x, int y);

int Test(int X, int Y) {
  if (X - X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent [misc-redundant-expression]
  if (X / X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X % X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent

  if (X & X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X | X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X ^ X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent

  if (X < X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X <= X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X > X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X >= X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent

  if (X && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent
  if (X || X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent

  if (X != (((X)))) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both side of operator are equivalent

  if (X + 1 == X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both side of operator are equivalent
  if (X + 1 != X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both side of operator are equivalent
  if (X + 1 <= X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both side of operator are equivalent
  if (X + 1 >= X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both side of operator are equivalent

  if ((X != 1 || Y != 1) && (X != 1 || Y != 1)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: both side of operator are equivalent
  if (P.a[X - P.x] != P.a[X - P.x]) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: both side of operator are equivalent

  if ((int)X < (int)X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both side of operator are equivalent

  if ( + "dummy" == + "dummy") return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both side of operator are equivalent
  if (L"abc" == L"abc") return 1;     
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both side of operator are equivalent

  if (foo(0) - 2 < foo(0) - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both side of operator are equivalent  
  if (foo(bar(0)) < (foo(bar((0))))) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both side of operator are equivalent  

  if (P1.x < P2.x && P1.x < P2.x) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both side of operator are equivalent  
  if (P2.a[P1.x + 2] < P2.x && P2.a[(P1.x) + (2)] < (P2.x)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: both side of operator are equivalent  

  return 0;
}

int Valid(int X, int Y) {
  if (X != Y) return 1;
  if (X == X + 0) return 1;
  if (P.x == P.y) return 1;
  if (P.a[P.x] < P.a[P.y]) return 1;
  if (P.a[0] < P.a[1]) return 1;

  if (P.a[0] < P.a[0ULL]) return 1;
  if (0 < 0ULL) return 1;
  if ((int)0 < (int)0ULL) return 1;

  if (++X != ++X) return 1;
  if (P.a[X]++ != P.a[X]++) return 1;
  if (P.a[X++] != P.a[X++]) return 1;

  if ("abc" == "ABC") return 1;
  if (foo(bar(0)) < (foo(bat(0, 1)))) return 1;
  return 0;
}

#define LT(x, y) (void)((x) < (y))

int TestMacro(int X, int Y) {
  LT(0, 0);
  LT(1, 0);
  LT(X, X);
  LT(X+1, X + 1);
}

int TestFalsePositive(int* A, int X, float F) {
  // Produced by bison.
  X = A[(2) - (2)];
  X = A['a' - 'a'];

  // Testing NaN.
  if (F != F && F == F) return 1;
  return 0;
}
