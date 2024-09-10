// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR: ![[annon_struct:.*]] = !cir.struct<struct  {!cir.int<s, 32>, !cir.array<!cir.int<s, 32> x 2>}>
// CIR: ![[sized_array:.*]] = !cir.struct<struct "sized_array" {!cir.int<s, 32>, !cir.array<!cir.int<s, 32> x 0>}

void escape(const void *);

// CIR-DAG: cir.global "private" internal dsolocal @T1._x = #cir.int<99> : !s8i
// LLVM-DAG: internal global i8 99

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
  // CIR-DAG: cir.global "private" constant internal @T2._a = #cir.const_struct<{#cir.int<2> : !s32i, #cir.const_array<[#cir.int<50> : !s32i, #cir.int<60> : !s32i]> : !cir.array<!s32i x 2>}>
  // LLVM-DAG: internal constant { i32, [2 x i32] } { i32 2, [2 x i32] [i32 50, i32 60] }
  const struct sized_array *A = ARRAY_PTR(50, 60);

  // CIR-DAG: cir.global "private" constant internal @T2._a.1 = #cir.const_struct<{#cir.int<3> : !s32i, #cir.const_array<[#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.int<30> : !s32i]> : !cir.array<!s32i x 3>}>
  // LLVM-DAG: internal constant { i32, [3 x i32] } { i32 3, [3 x i32] [i32 10, i32 20, i32 30] }
  struct outer X = {ARRAY_PTR(10, 20, 30)};

  // CIR-DAG: %[[T2A:.*]] = cir.get_global @T2._a : !cir.ptr<![[annon_struct]]>
  // CIR-DAG: cir.cast(bitcast, %[[T2A]] : !cir.ptr<![[annon_struct]]>), !cir.ptr<![[sized_array]]>
  escape(A);
  escape(&X);
}
