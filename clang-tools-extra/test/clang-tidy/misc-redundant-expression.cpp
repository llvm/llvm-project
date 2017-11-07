// RUN: %check_clang_tidy %s misc-redundant-expression %t -- -- -std=c++11

typedef __INT64_TYPE__ I64;

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

int TestSimpleEquivalent(int X, int Y) {
  if (X - X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent [misc-redundant-expression]
  if (X / X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X % X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X & X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X | X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X ^ X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X < X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X <= X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X > X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X >= X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X || X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X != (((X)))) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X + 1 == X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X + 1 != X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X + 1 <= X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X + 1 >= X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent

  if ((X != 1 || Y != 1) && (X != 1 || Y != 1)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: both sides of operator are equivalent
  if (P.a[X - P.x] != P.a[X - P.x]) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: both sides of operator are equivalent

  if ((int)X < (int)X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent
  if (int(X) < int(X)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent

  if ( + "dummy" == + "dummy") return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent
  if (L"abc" == L"abc") return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent

  if (foo(0) - 2 < foo(0) - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent
  if (foo(bar(0)) < (foo(bar((0))))) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both sides of operator are equivalent

  if (P1.x < P2.x && P1.x < P2.x) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both sides of operator are equivalent
  if (P2.a[P1.x + 2] < P2.x && P2.a[(P1.x) + (2)] < (P2.x)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: both sides of operator are equivalent

  return 0;
}

int Valid(int X, int Y) {
  if (X != Y) return 1;
  if (X == Y + 0) return 1;
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

#define COND_OP_MACRO 9
#define COND_OP_OTHER_MACRO 9
int TestConditional(int x, int y) {
  int k = 0;
  k += (y < 0) ? x : x;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'true' and 'false' expressions are equivalent
  k += (y < 0) ? x + 1 : x + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'true' and 'false' expressions are equivalent
  k += (y < 0) ? COND_OP_MACRO : COND_OP_MACRO;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: 'true' and 'false' expressions are equivalent

  // Do not match for conditional operators with a macro and a const.
  k += (y < 0) ? COND_OP_MACRO : 9;
  // Do not match for conditional operators with expressions from different macros.
  k += (y < 0) ? COND_OP_MACRO : COND_OP_OTHER_MACRO;
  return k;
}
#undef COND_OP_MACRO
#undef COND_OP_OTHER_MACRO

struct MyStruct {
  int x;
} Q;

bool operator==(const MyStruct& lhs, const MyStruct& rhs) { return lhs.x == rhs.x; }

bool TestOperator(MyStruct& S) {
  if (S == Q) return false;
  if (S == S) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of overloaded operator are equivalent
}

#define LT(x, y) (void)((x) < (y))
#define COND(x, y, z) ((x)?(y):(z))
#define EQUALS(x, y) (x) == (y)

int TestMacro(int X, int Y) {
  LT(0, 0);
  LT(1, 0);
  LT(X, X);
  LT(X+1, X + 1);
  COND(X < Y, X, X);
  EQUALS(Q, Q);
  return 0;
}

int TestFalsePositive(int* A, int X, float F) {
  // Produced by bison.
  X = A[(2) - (2)];
  X = A['a' - 'a'];

  // Testing NaN.
  if (F != F && F == F) return 1;
  return 0;
}

int TestBannedMacros() {
#define EAGAIN 3
#define NOT_EAGAIN 3
  if (EAGAIN == 0 | EAGAIN == 0) return 0;
  if (NOT_EAGAIN == 0 | NOT_EAGAIN == 0) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: both sides of operator are equivalent
  return 0;
}

struct MyClass {
static const int Value = 42;
};
template <typename T, typename U>
void TemplateCheck() {
  static_assert(T::Value == U::Value, "should be identical");
  static_assert(T::Value == T::Value, "should be identical");
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: both sides of overloaded operator are equivalent
}
void TestTemplate() { TemplateCheck<MyClass, MyClass>(); }

int TestArithmetic(int X, int Y) {
  if (X + 1 == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 != X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true
  if (X - 1 == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X - 1 != X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1LL == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X + 1ULL == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always false

  if (X == X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always false
  if (X != X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true
  if (X == X - 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always false
  if (X != X - 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true

  if (X != X - 1U) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true
  if (X != X - 1LL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true

  if ((X+X) != (X+X) - 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X + 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 != X + 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X - 1 == X - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X - 1 != X - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X - -1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true
  if (X + 1 != X - -1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 == X - -2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 != X - -2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X - (~0)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true
  if (X + 1 == X - (~0U)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X - (~0ULL)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  // Should not match.
  if (X + 0.5 == X) return 1;
  if (X + 1 == Y) return 1;
  if (X + 1 == Y + 1) return 1;
  if (X + 1 == Y + 2) return 1;

  return 0;
}

int TestBitwise(int X, int Y) {

  if ((X & 0xFF) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((X & 0xFF) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true
  if ((X | 0xFF) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((X | 0xFF) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true

  if ((X | 0xFFULL) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: logical expression is always true
  if ((X | 0xFF) != 0xF00ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true

  if ((0xFF & X) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((0xFF & X) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true
  if ((0xFF & X) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((0xFF & X) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true

  if ((0xFFLL & X) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: logical expression is always false
  if ((0xFF & X) == 0xF00ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false

  return 0;
}

int TestLogical(int X, int Y){
#define CONFIG 0
  if (CONFIG && X) return 1; // OK, consts from macros are considered intentional
#undef CONFIG
#define CONFIG 1
  if (CONFIG || X) return 1;
#undef CONFIG

  if (X == 10 && X != 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X == 10 && (X != 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X == 10 && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (!(X != 10) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false

  if (X == 10ULL && X != 10ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if (!(X != 10U) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: logical expression is always false
  if (!(X != 10LL) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: logical expression is always false
  if (!(X != 10ULL) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: logical expression is always false

  if (X == 0 && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X != 0 && !X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X && !X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always false

  if (X && !!X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: equivalent expression on both sides of logical operator
  if (X != 0 && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator
  if (X != 0 && !!X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator
  if (X == 0 && !X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator

  // Should not match.
  if (X == 10 && Y == 10) return 1;
  if (X != 10 && X != 12) return 1;
  if (X == 10 || X == 12) return 1;
  if (!X && !Y) return 1;
  if (!X && Y) return 1;
  if (!X && Y == 0) return 1;
  if (X == 10 && Y != 10) return 1;
}

int TestRelational(int X, int Y) {
  if (X == 10 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X == 10 && X < 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X < 10 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X <= 10 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X < 10 && X >= 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X < 10 && X == 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false

  if (X > 5 && X <= 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X > -5 && X <= -5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false

  if (X < 10 || X >= 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always true
  if (X <= 10 || X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (X <= 10 || X >= 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (X != 7 || X != 14) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always true
  if (X == 7 || X != 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 7 || X == 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always true

  if (X < 7 && X < 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X < 7 && X < 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X < 7 && X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant

  if (X < 7 && X <= 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X < 7 && X <= 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: equivalent expression on both sides of logical operator
  if (X < 7 && X <= 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant
  if (X < 7 && X <= 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant

  if (X <= 7 && X < 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X < 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator

  if (X >= 7 && X > 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator
  if (X >= 7 && X > 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X >= 7 && X > 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X <= 7 && X <= 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X <= 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X <= 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent
  if (X <= 7 && X <= 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: expression is redundant

  if (X == 11 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: expression is redundant
  if (X == 11 && X < 12) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: expression is redundant
  if (X > 10 && X == 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X < 12 && X == 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X != 11 && X == 42) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X > 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X < 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X > 14) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X < 7 || X < 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant
  if (X < 7 || X < 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X < 7 || X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X > 7 || X > 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X > 7 || X > 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X > 7 || X > 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant

  // Should not match.
  if (X < 10 || X > 12) return 1;
  if (X > 10 && X < 12) return 1;
  if (X < 10 || X >= 12) return 1;
  if (X > 10 && X <= 12) return 1;
  if (X <= 10 || X > 12) return 1;
  if (X >= 10 && X < 12) return 1;
  if (X <= 10 || X >= 12) return 1;
  if (X >= 10 && X <= 12) return 1;
  if (X >= 10 && X <= 11) return 1;
  if (X >= 10 && X < 11) return 1;
  if (X > 10 && X <= 11) return 1;
  if (X > 10 && X != 11) return 1;
  if (X >= 10 && X <= 10) return 1;
  if (X <= 10 && X >= 10) return 1;
  if (X < 0 || X > 0) return 1;
}

int TestRelationalMacros(int X){
#define SOME_MACRO 3
#define SOME_MACRO_SAME_VALUE 3
#define SOME_OTHER_MACRO 9
  // Do not match for redundant relational macro expressions that can be
  // considered intentional, and for some particular values, non redundant.

  // Test cases for expressions with the same macro on both sides.
  if (X < SOME_MACRO && X > SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always false
  if (X < SOME_MACRO && X == SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always false
  if (X < SOME_MACRO || X >= SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always true
  if (X <= SOME_MACRO || X > SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: logical expression is always true
  if (X != SOME_MACRO && X > SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != SOME_MACRO && X < SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  // Test cases for two different macros.
  if (X < SOME_MACRO && X > SOME_OTHER_MACRO) return 1;
  if (X != SOME_MACRO && X >= SOME_OTHER_MACRO) return 1;
  if (X != SOME_MACRO && X != SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO || X == SOME_MACRO_SAME_VALUE) return 1;
  if (X == SOME_MACRO || X <= SOME_MACRO_SAME_VALUE) return 1;
  if (X == SOME_MACRO || X > SOME_MACRO_SAME_VALUE) return 1;
  if (X < SOME_MACRO && X <= SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO && X > SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO && X != SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO && X != SOME_MACRO_SAME_VALUE) return 1;
  if (X == SOME_MACRO_SAME_VALUE && X == SOME_MACRO ) return 1;

  // Test cases for a macro and a const.
  if (X < SOME_MACRO && X > 9) return 1;
  if (X != SOME_MACRO && X >= 9) return 1;
  if (X != SOME_MACRO && X != 9) return 1;
  if (X == SOME_MACRO || X == 3) return 1;
  if (X == SOME_MACRO || X <= 3) return 1;
  if (X < SOME_MACRO && X <= 9) return 1;
  if (X == SOME_MACRO && X != 9) return 1;
  if (X == SOME_MACRO && X == 9) return 1;

#undef SOME_OTHER_MACRO
#undef SOME_MACRO_SAME_VALUE
#undef SOME_MACRO
  return 0;
}

int TestValidExpression(int X) {
  if (X - 1 == 1 - X) return 1;
  if (2 * X == X) return 1;
  if ((X << 1) == X) return 1;

  return 0;
}

enum Color { Red, Yellow, Green };
int TestRelationalWithEnum(enum Color C) {
  if (C == Red && C == Yellow) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always false
  if (C == Red && C != Red) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always false
  if (C != Red || C != Yellow) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always true

  // Should not match.
  if (C == Red || C == Yellow) return 1;
  if (C != Red && C != Yellow) return 1;

  return 0;
}

template<class T>
int TestRelationalTemplated(int X) {
  // This test causes a corner case with |isIntegerConstantExpr| where the type
  // is dependent. There is an assert failing when evaluating
  // sizeof(<incomplet-type>).
  if (sizeof(T) == 4 || sizeof(T) == 8) return 1;

  if (X + 0 == -X) return 1;
  if (X + 0 < X) return 1;

  return 0;
}

int TestWithSignedUnsigned(int X) {
  if (X + 1 == X + 1ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if ((X & 0xFFU) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: logical expression is always false

  if ((X & 0xFF) == 0xF00U) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false

  if ((X & 0xFFU) == 0xF00U) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: logical expression is always false

  return 0;
}

int TestWithLong(int X, I64 Y) {
  if (X + 0ULL == -X) return 1;
  if (Y + 0 == -Y) return 1;
  if (Y <= 10 && X >= 10LL) return 1;
  if (Y <= 10 && X >= 10ULL) return 1;
  if (X <= 10 || X > 12LL) return 1;
  if (X <= 10 || X > 12ULL) return 1;
  if (Y <= 10 || Y > 12) return 1;

  return 0;
}

int TestWithMinMaxInt(int X) {
  if (X <= X + 0xFFFFFFFFU) return 1;
  if (X <= X + 0x7FFFFFFF) return 1;
  if (X <= X + 0x80000000) return 1;

  if (X <= 0xFFFFFFFFU && X > 0) return 1;
  if (X <= 0xFFFFFFFFU && X > 0U) return 1;

  if (X + 0x80000000 == X - 0x80000000) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always true

  if (X > 0x7FFFFFFF || X < ((-0x7FFFFFFF)-1)) return 1;
  if (X <= 0x7FFFFFFF && X >= ((-0x7FFFFFFF)-1)) return 1;

  return 0;
}
