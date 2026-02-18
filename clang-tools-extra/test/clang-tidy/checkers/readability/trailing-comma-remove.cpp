// RUN: %check_clang_tidy -std=c++98-or-later %s readability-trailing-comma %t -- \
// RUN:   -config='{CheckOptions: {readability-trailing-comma.SingleLineCommaPolicy: Remove, readability-trailing-comma.MultiLineCommaPolicy: Remove}}'

struct S { int x, y; };

enum E1 {
  A,
  B,
  C,
};
// CHECK-MESSAGES: :[[@LINE-2]]:4: warning: enum should not have a trailing comma [readability-trailing-comma]
// CHECK-FIXES: enum E1 {
// CHECK-FIXES-NEXT:   A,
// CHECK-FIXES-NEXT:   B,
// CHECK-FIXES-NEXT:   C
// CHECK-FIXES-NEXT: };

enum E2 {
  V = 1,
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: enum should not have a trailing comma
// CHECK-FIXES: enum E2 {
// CHECK-FIXES-NEXT:   V = 1
// CHECK-FIXES-NEXT: };

enum SingleLine { A1, B1, C1, };
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: enum should not have a trailing comma
// CHECK-FIXES: enum SingleLine { A1, B1, C1 };

enum E3 {
  P,
  Q
};
enum Empty {};

void f() {
  // Multi-line init lists with trailing commas - should warn to remove
  int a[] = {
    1,
    2,
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int a[] = {
  // CHECK-FIXES-NEXT:     1,
  // CHECK-FIXES-NEXT:     2
  // CHECK-FIXES-NEXT:   };

  S s = {
    1,
    2,
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: S s = {
  // CHECK-FIXES-NEXT:     1,
  // CHECK-FIXES-NEXT:     2
  // CHECK-FIXES-NEXT:   };

  int b[] = {1, 2, 3,};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int b[] = {1, 2, 3};
  S s2 = {1, 2,};
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: S s2 = {1, 2};

  int c[] = {
    1,
    2
  };
  int d[] = {};
}

struct N { S a, b; };
void nested() {
  N n = {{3, 4,},};
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: N n = {{[{][{]3, 4[}][}]}};
  N n2 = {{3, 4}};
}
