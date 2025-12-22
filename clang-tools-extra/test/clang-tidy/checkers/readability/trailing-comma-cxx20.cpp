// RUN: %check_clang_tidy -std=c++20-or-later %s readability-trailing-comma %t -- \
// RUN:   -config='{CheckOptions: {readability-trailing-comma.InitListThreshold: 1}}'

struct S { int x, y; };

void f() {
  S s1 = {.x = 1, .y = 2};
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: initializer list should have a trailing comma
  // CHECK-FIXES: S s1 = {.x = 1, .y = 2,};

  S s2 = {.x = 1};
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: initializer list should have a trailing comma
  // CHECK-FIXES: S s2 = {.x = 1,};

  int a[3] = {[0] = 1, [2] = 3};
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: initializer list should have a trailing comma
  // CHECK-FIXES: int a[3] = {[0] = 1, [2] = 3,};

  // No warnings
  S s3 = {.x = 1, .y = 2,};
}

struct N { S a, b; };

void nested() {
  N n = {.a = {.x = 1, .y = 2}, .b = {.x = 3, .y = 4}};
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: initializer list should have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:53: warning: initializer list should have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-3]]:54: warning: initializer list should have a trailing comma

  // No warning
  N n2 = {.a = {.x = 1, .y = 2,}, .b = {.x = 3, .y = 4,},};
}

struct WithArray {
  int values[3];
  int count;
};

void with_array() {
  WithArray w1 = {.values = {1, 2, 3}, .count = 3};
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: initializer list should have a trailing comma [readability-trailing-comma]
  // CHECK-MESSAGES: :[[@LINE-2]]:50: warning: initializer list should have a trailing comma [readability-trailing-comma]

  WithArray w2 = {.values = {1, 2, 3,}, .count = 3,};
}
