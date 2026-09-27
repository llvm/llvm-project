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

struct AnonUnion {
  int x;
  union { struct { int a; int b; }; };
};

void anonymous_union_members() {
  AnonUnion w1 = {
    .x = 1,
    .a = 2,
    .b = 3,
  };

  AnonUnion w2 = {
    .x = 1,
    .a = 2,
    .b = 3
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: initializer list should have a trailing comma
  // CHECK-FIXES: AnonUnion w2 = {
  // CHECK-FIXES-NEXT:     .x = 1,
  // CHECK-FIXES-NEXT:     .a = 2,
  // CHECK-FIXES-NEXT:     .b = 3,
  // CHECK-FIXES-NEXT:   };
}

struct Inner { int v; };
struct Nested { Inner x; Inner y; };

void nested_designator() {
  Nested n1 = {
    .x = {.v = 1},
    .y.v = 2,
  };

  Nested n2 = {
    .x = {.v = 1},
    .y.v = 2
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: initializer list should have a trailing comma
  // CHECK-FIXES: Nested n2 = {
  // CHECK-FIXES-NEXT:     .x = {.v = 1},
  // CHECK-FIXES-NEXT:     .y.v = 2,
  // CHECK-FIXES-NEXT:   };

  Nested n3 = {
    .x = {.v = 1},
    .y = {.v = 2,},
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: Nested n3 = {
  // CHECK-FIXES-NEXT:     .x = {.v = 1},
  // CHECK-FIXES-NEXT:     .y = {.v = 2},
  // CHECK-FIXES-NEXT:   };
}

struct AnonStruct {
  int x;
  struct { int p; int q; };
};

void anonymous_struct_members() {
  AnonStruct as1 = {
    .x = 1,
    .p = 2,
    .q = 3,
  };

  AnonStruct as2 = { .x = 1, .p = 2, .q = 3, };
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: initializer list should not have a trailing comma
  // CHECK-FIXES: AnonStruct as2 = { .x = 1, .p = 2, .q = 3 };
}

struct Deep { int c; };
struct Mid { Deep b; };
struct Top { Mid a; };

void multi_level_designator() {
  Top t1 = {
    .a.b.c = 1,
  };

  Top t2 = {
    .a.b.c = 1
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: initializer list should have a trailing comma
  // CHECK-FIXES: Top t2 = {
  // CHECK-FIXES-NEXT:     .a.b.c = 1,
  // CHECK-FIXES-NEXT:   };
}

struct TwoFields { int v; int w; };
struct Holder { TwoFields y; };

void repeated_subobject_designator() {
  Holder h1 = {
    .y.v = 1,
    .y.w = 2,
  };
}

struct WithArrayField { int vals[3]; int n; };

void array_designator() {
  WithArrayField wa1 = {
    .vals[0] = 1,
    .n = 1,
  };

  WithArrayField wa2 = {
    .vals[0] = 1,
    .n = 1
  };
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: initializer list should have a trailing comma
  // CHECK-FIXES: WithArrayField wa2 = {
  // CHECK-FIXES-NEXT:     .vals[0] = 1,
  // CHECK-FIXES-NEXT:     .n = 1,
  // CHECK-FIXES-NEXT:   };
}
