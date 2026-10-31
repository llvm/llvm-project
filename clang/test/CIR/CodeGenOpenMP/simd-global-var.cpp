// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int g;

// CIR: cir.global external @g = #cir.int<0> : !s32i
// LLVM: @g = global i32 0

int global_ref(int v) {
  g = v;
  return g;
}

// CIR: cir.func{{.*}} @_Z10global_refi
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %{{.*}}, %[[G]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[G1:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   %{{.*}} = cir.load{{.*}} %[[G1]] : !cir.ptr<!s32i>, !s32i

// LLVM: define{{.*}} i32 @_Z10global_refi
// LLVM:   store i32 %{{.*}}, ptr @g
// LLVM:   %{{.*}} = load i32, ptr @g
