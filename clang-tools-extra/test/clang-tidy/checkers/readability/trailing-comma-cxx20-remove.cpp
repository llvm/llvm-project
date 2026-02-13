// RUN: %check_clang_tidy -std=c++20-or-later %s readability-trailing-comma %t -- \
// RUN:   -config='{CheckOptions: {readability-trailing-comma.SingleLineCommaPolicy: Remove, readability-trailing-comma.MultiLineCommaPolicy: Remove}}'

struct S { int x, y; };

void f() {
  S s1 = {
    .x = 1,
    .y = 2,
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: S s1 = {
  // CHECK-FIXES-NEXT:     .x = 1,
  // CHECK-FIXES-NEXT:     .y = 2
  // CHECK-FIXES-NEXT:   };

  int a[3] = {
    [0] = 1,
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: int a[3] = {
  // CHECK-FIXES-NEXT:     [0] = 1
  // CHECK-FIXES-NEXT:   };

  S s2 = {.x = 1, .y = 2,};
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: S s2 = {.x = 1, .y = 2};

  S s3 = {
    .x = 1,
    .y = 2
  };
}

struct N { S a, b; };

void nested() { 
  N n1 = {.a = {.x = 1, .y = 2}, .b = {.x = 3, .y = 4,},};
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:56: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: N n1 = {.a = {.x = 1, .y = 2}, .b = {.x = 3, .y = 4}};

  N n2 = {
    .a = {.x = 1, .y = 2},
    .b = {.x = 3, .y = 4}
  };
}

struct WithArray {
  int values[3];
  int count;
};

void with_array() {
  WithArray w2 = {.values = {1, 2, 3,}, .count = 3,};
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: initializer list should not have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-2]]:51: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: WithArray w2 = {.values = {1, 2, 3}, .count = 3};

  WithArray w3 = {
    .values = {1, 2, 3},
    .count = 3
  };
}
