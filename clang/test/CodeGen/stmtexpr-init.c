// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void escape(const void *);

// CHECK-DAG: internal global i8 99

void T1(void) {
  const char *x[1] = {({static char _x = 99; &_x; })};
  escape(x);
}

struct sized_array {
  int count;
  int entries[];
};

#define N_ARGS(...) (sizeof((int[]){__VA_ARGS__}) / sizeof(int))

#define ARRAY_PTR(...) ({                                                    \
  static const struct sized_array _a = {N_ARGS(__VA_ARGS__), {__VA_ARGS__}}; \
  &_a;                                                                       \
})

struct outer {
  const struct sized_array *a;
};

void T2(void) {
  // CHECK-DAG: internal constant { i32, [2 x i32] } { i32 2, [2 x i32] [i32 50, i32 60] }
  const struct sized_array *A = ARRAY_PTR(50, 60);

  // CHECK-DAG: internal constant { i32, [3 x i32] } { i32 3, [3 x i32] [i32 10, i32 20, i32 30] }
  struct outer X = {ARRAY_PTR(10, 20, 30)};

  escape(A);
  escape(&X);
}
