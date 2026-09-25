// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

const int c0 __attribute__((retain)) = 42;

__attribute__((retain)) int g0;
int g1 __attribute__((retain));
__attribute__((used, retain)) static int g3;

void __attribute__((retain)) f0(void) {}
static void __attribute__((used, retain)) f2(void) {}

// CIR: cir.global "private" appending @llvm.used = #cir.const_array<[#cir.global_view<@c0> : !cir.ptr<!void>, #cir.global_view<@f0> : !cir.ptr<!void>, #cir.global_view<@f2> : !cir.ptr<!void>, #cir.global_view<@g0> : !cir.ptr<!void>, #cir.global_view<@g1> : !cir.ptr<!void>, #cir.global_view<@g3> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 6>
// CIR: cir.global "private" appending @llvm.compiler.used = #cir.const_array<[#cir.global_view<@f2> : !cir.ptr<!void>, #cir.global_view<@g3> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 2>

// LLVM: @llvm.used = appending global [6 x ptr] [ptr @c0, ptr @f0, ptr @f2, ptr @g0, ptr @g1, ptr @g3], section "llvm.metadata"
// LLVM: @llvm.compiler.used = appending global [2 x ptr] [ptr @f2, ptr @g3], section "llvm.metadata"
