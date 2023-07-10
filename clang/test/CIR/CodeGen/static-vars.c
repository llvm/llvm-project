// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void func1(void) {
  // Should lower default-initialized static vars.
  static int i;
  // CHECK-DAG: cir.global "private" internal @func1.i = #cir.int<0> : !s32i

  // Should lower constant-initialized static vars.
  static int j = 1;
  // CHECK-DAG: cir.global "private" internal @func1.j = #cir.int<1> : !s32i

  // Should properly shadow static vars in nested scopes.
  {
    static int j = 2;
    // CHECK-DAG: cir.global "private" internal @func1.j.1 = #cir.int<2> : !s32i
  }
  {
    static int j = 3;
    // CHECK-DAG: cir.global "private" internal @func1.j.2 = #cir.int<3> : !s32i
  }

  // Should lower basic static vars arithmetics.
  j++;
  // CHECK-DAG: %[[#V2:]] = cir.get_global @func1.j : cir.ptr <!s32i>
  // CHECK-DAG: %[[#V3:]] = cir.load %[[#V2]] : cir.ptr <!s32i>, !s32i
  // CHECK-DAG: %[[#V4:]] = cir.unary(inc, %[[#V3]]) : !s32i, !s32i
  // CHECK-DAG: cir.store %[[#V4]], %[[#V2]] : !s32i, cir.ptr <!s32i>
}

// Should shadow static vars on different functions.
void func2(void) {
  static char i;
  // CHECK-DAG: cir.global "private" internal @func2.i = #cir.int<0> : !s8i
  static float j;
  // CHECK-DAG: cir.global "private" internal @func2.j = 0.000000e+00 : f32
}
