// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

int g0 __attribute__((used));

static void __attribute__((used)) f0(void) {
}

__attribute__((used)) int a0;
void pr27535(void) { (void)a0; }

// CIR: cir.global "private" appending @llvm.compiler.used = #cir.const_array<[#cir.global_view<@f0> : !cir.ptr<!void>, #cir.global_view<@g0> : !cir.ptr<!void>, #cir.global_view<@a0> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 3>
// LLVM: @llvm.compiler.used = appending global [3 x ptr] [ptr @f0, ptr @g0, ptr @a0], section "llvm.metadata"
