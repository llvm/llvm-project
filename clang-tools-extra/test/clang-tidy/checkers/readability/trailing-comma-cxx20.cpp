// RUN: %check_clang_tidy -std=c++20-or-later %s readability-trailing-comma %t

struct S { int x, y; };

void f() {
  S s1 = {
    .x = 1,
    .y = 2
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: initializer list should have a trailing comma
  // CHECK-FIXES: S s1 = {
  // CHECK-FIXES-NEXT:     .x = 1,
  // CHECK-FIXES-NEXT:     .y = 2,
  // CHECK-FIXES-NEXT:   };

  int a[3] = {
    [0] = 1
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: initializer list should have a trailing comma
  // CHECK-FIXES: int a[3] = {
  // CHECK-FIXES-NEXT:     [0] = 1,
  // CHECK-FIXES-NEXT:   };

  S s2 = {.x = 1, .y = 2};
  S s3 = {.x = 1};

  S s4 = {
    .x = 1,
  };
}

struct N { S a, b; };

void nested() {
  N n = {
    .a = {.x = 1, .y = 2},
    .b = {
      .x = 3,
      .y = 4
    }
  };
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: initializer list should have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-3]]:6: warning: initializer list should have a trailing comma
  // CHECK-FIXES: N n = {
  // CHECK-FIXES-NEXT:    .a = {.x = 1, .y = 2},
  // CHECK-FIXES-NEXT:    .b = {
  // CHECK-FIXES-NEXT:      .x = 3,
  // CHECK-FIXES-NEXT:      .y = 4,
  // CHECK-FIXES-NEXT:    },
  // CHECK-FIXES-NEXT:   };

  N n2 = {.a = {.x = 1, .y = 2}, .b = {.x = 3, .y = 4}};

  N n3 = {
    .a = {.x = 1, .y = 2},
    .b = {
      .x = 3,
      .y = 4,
    },
  };
}

struct WithArray {
  int values[3];
  int count;
};

void with_array() {
  WithArray w1 = {
    .values = {1, 2,
      3
    },
    .count = 3
  };
  // CHECK-MESSAGES: :[[@LINE-4]]:8: warning: initializer list should have a trailing comma
  // CHECK-MESSAGES: :[[@LINE-3]]:15: warning: initializer list should have a trailing comma
  // CHECK-FIXES: WithArray w1 = {
  // CHECK-FIXES-NEXT:    .values = {1, 2,
  // CHECK-FIXES-NEXT:      3,
  // CHECK-FIXES-NEXT:    },
  // CHECK-FIXES-NEXT:    .count = 3,
  // CHECK-FIXES-NEXT:   };

  WithArray w2 = {.values = {1, 2, 3}, .count = 3};
  WithArray w3 = {
    .values = {1, 2, 3},
    .count = 3,
  };
}
