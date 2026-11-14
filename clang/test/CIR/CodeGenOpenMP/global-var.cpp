// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-offload.cir
// RUN: FileCheck --input-file=%t-offload.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int g;
extern int e;
int *use_e = &e;

// CIR: cir.global external @g = #cir.int<0> : !s32i
// CIR: cir.global external @use_e = #cir.global_view<@e> : !cir.ptr<!s32i>
// CIR: cir.global "private" external @e : !s32i

// LLVM: @g = global i32 0
// LLVM: @use_e = global ptr @e
// LLVM: @e = external global i32

int global_ref(int v) {
  g = v;
  return g + e;
}

// CIR: cir.func{{.*}} @_Z10global_refi
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %{{.*}}, %[[G]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[G1:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   %[[G_VAL:.*]] = cir.load{{.*}} %[[G1]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[E:.*]] = cir.get_global @e : !cir.ptr<!s32i>
// CIR:   %[[E_VAL:.*]] = cir.load{{.*}} %[[E]] : !cir.ptr<!s32i>, !s32i
// CIR:   %{{.*}} = cir.add nsw %[[G_VAL]], %[[E_VAL]] : !s32i

// LLVM: define{{.*}} i32 @_Z10global_refi
// LLVM:   store i32 %{{.*}}, ptr @g
// LLVM:   %[[G:.*]] = load i32, ptr @g
// LLVM:   %[[E:.*]] = load i32, ptr @e
// LLVM:   %{{.*}} = add nsw i32 %[[G]], %[[E]]
