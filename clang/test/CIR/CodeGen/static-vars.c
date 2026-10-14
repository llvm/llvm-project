// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void func1(void) {
  // Should lower default-initialized static vars.
  static int i;
  // CIR-DAG: cir.global "private" internal dso_local @func1.i = #cir.int<0> : !s32i

  // Should lower constant-initialized static vars.
  static int j = 1;
  // CIR-DAG: cir.global "private" internal dso_local @func1.j = #cir.int<1> : !s32i

  // Should properly shadow static vars in nested scopes.
  {
    static int j = 2;
    // CIR-DAG: cir.global "private" internal dso_local @func1.j.1 = #cir.int<2> : !s32i
  }
  {
    static int j = 3;
    // CIR-DAG: cir.global "private" internal dso_local @func1.j.2 = #cir.int<3> : !s32i
  }

  // Should lower basic static vars arithmetics.
  j++;
  // CIR-DAG: %[[#V2:]] = cir.get_global @func1.j : !cir.ptr<!s32i>
  // CIR-DAG: %[[#V3:]] = cir.load{{.*}} %[[#V2]] : !cir.ptr<!s32i>, !s32i
  // CIR-DAG: %[[#V4:]] = cir.inc nsw %[[#V3]] : !s32i
  // CIR-DAG: cir.store{{.*}} %[[#V4]], %[[#V2]] : !s32i, !cir.ptr<!s32i>
}

// Should shadow static vars on different functions.
void func2(void) {
  static char i;
  // CIR-DAG: cir.global "private" internal dso_local @func2.i = #cir.int<0> : !s8i
  static float j;
  // CIR-DAG: cir.global "private" internal dso_local @func2.j = #cir.fp<0.000000e+00> : !cir.float
}

// LLVM-DAG: @func1.i = internal global i32 0
// LLVM-DAG: @func1.j = internal global i32 1
// LLVM-DAG: @func1.j.1 = internal global i32 2
// LLVM-DAG: @func1.j.2 = internal global i32 3
// LLVM-DAG: @func2.i = internal global i8 0
// LLVM-DAG: @func2.j = internal global float 0.000000e+00

// LLVM: define {{.*}}void @func1()
// LLVM:   load i32, ptr @func1.j
// LLVM:   add nsw i32 %{{.*}}, 1
// LLVM:   store i32 %{{.*}}, ptr @func1.j

// LLVM: define {{.*}}void @func2()

// OGCG-DAG: @func1.i = internal global i32 0
// OGCG-DAG: @func1.j = internal global i32 1
// OGCG-DAG: @func1.j.1 = internal global i32 2
// OGCG-DAG: @func1.j.2 = internal global i32 3
// OGCG-DAG: @func2.i = internal global i8 0
// OGCG-DAG: @func2.j = internal global float 0.000000e+00

// OGCG: define {{.*}}void @func1()
// OGCG:   load i32, ptr @func1.j
// OGCG:   add nsw i32 %{{.*}}, 1
// OGCG:   store i32 %{{.*}}, ptr @func1.j

// OGCG: define {{.*}}void @func2()
