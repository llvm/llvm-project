// RUN: %check_clang_tidy -std=c++11-or-later %s readability-trailing-comma %t -- \
// RUN:   -config='{CheckOptions: {readability-trailing-comma.CommaPolicy: Remove, \
// RUN:             readability-trailing-comma.InitListThreshold: 1}}'

struct S { int x, y; };

enum E1 { A, B, C, };
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: enum should not have a trailing comma [readability-trailing-comma]
// CHECK-FIXES: enum E1 { A, B, C };

enum class E2 { X, Y, };
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: enum should not have a trailing comma
// CHECK-FIXES: enum class E2 { X, Y };

enum E3 { V = 1, };
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: enum should not have a trailing comma
// CHECK-FIXES: enum E3 { V = 1 };

enum E4 { P, Q };
enum Empty {};

void f() {
  int a[] = {1, 2,};
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int a[] = {1, 2};

  S s = {1, 2,};
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: S s = {1, 2};

  int b[] = {1,};
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int b[] = {1};

  int c[] = {1, 2};
  S s2 = {1, 2};
  int d[] = {};
}

struct N { S a, b; };

void nested() {
  N n = {{1, 2,}, {3, 4,},};
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:24: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-3]]:26: warning: initializer list should not have a trailing comma

  N n2 = {{1, 2}, {3, 4}};
}
