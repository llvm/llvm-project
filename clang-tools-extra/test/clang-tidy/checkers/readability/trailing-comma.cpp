// RUN: %check_clang_tidy -std=c++11-or-later %s readability-trailing-comma %t

struct S { int x, y; };

enum E1 {
  A,
  B,
  C
};
// CHECK-MESSAGES: :[[@LINE-2]]:4: warning: enum should have a trailing comma [readability-trailing-comma]
// CHECK-FIXES: enum E1 {
// CHECK-FIXES-NEXT:   A,
// CHECK-FIXES-NEXT:   B,
// CHECK-FIXES-NEXT:   C,
// CHECK-FIXES-NEXT: };

enum E2 {
  V = 1
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: enum should have a trailing comma
// CHECK-FIXES: enum E2 {
// CHECK-FIXES-NEXT:   V = 1,
// CHECK-FIXES-NEXT: };

enum SingleLine { A1, B1, C1 };
enum class SingleLine2 { X1, Y1, };
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: enum should not have a trailing comma
// CHECK-FIXES: enum class SingleLine2 { X1, Y1 };

enum E3 {
  P,
  Q,
};
enum Empty {};

void f() {
  int a[] = {
    1
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should have a trailing comma
  // CHECK-FIXES: int a[] = {
  // CHECK-FIXES-NEXT:     1,
  // CHECK-FIXES-NEXT:   };

  S s = {
    1,
    2
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should have a trailing comma
  // CHECK-FIXES: S s = {
  // CHECK-FIXES-NEXT:     1,
  // CHECK-FIXES-NEXT:     2,
  // CHECK-FIXES-NEXT:   };

  int b[] = {1, 2, 3};
  S s2 = {1, 2};

  int e[] = {1, 2, 3,};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int e[] = {1, 2, 3};

  int c[] = {
    1,
    2,
  };
  int d[] = {};
}

struct N { S a, b; };
void nested() {
  N n = {{1, 2}, {3, 4}};
  N n2 = {{3, 4,},};
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: N n2 = {{[{][{]3, 4[}][}]}};
}

void nestedMultiLine() {
  N n = {
    {1, 2},
    {3, 4}
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: initializer list should have a trailing comma
  // CHECK-FIXES: N n = {
  // CHECK-FIXES-NEXT:     {1, 2},
  // CHECK-FIXES-NEXT:     {3, 4},
  // CHECK-FIXES-NEXT:   };

  N n2 = {
    {1, 2},
    {3, 4},
  };
}

// Macros are ignored
#define ENUM(n, a, b) enum n { a, b }
#define INIT {1, 2}

ENUM(E1M, Xm, Ym);
int macroArr[] = INIT;

// Template pack expansions - no warnings
template <typename T, typename... Ts>
struct Pack {
  int values[sizeof...(Ts) + 1] = {sizeof(T), sizeof(Ts)...};
};

Pack<int> single;
Pack<int, double> two;
Pack<int, double, char> three;

template <typename... Ts>
struct PackSingle {
  int values[sizeof...(Ts)] = {sizeof(Ts)...};
};

PackSingle<int> p1;
PackSingle<int, double, char> p3;
