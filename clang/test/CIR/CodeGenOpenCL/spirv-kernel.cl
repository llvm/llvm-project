// RUN: %clang_cc1 %s -fclangir -emit-cir -triple spirv64-unknown-unknown -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

// RUN: %clang_cc1 %s -fclangir -emit-llvm -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// RUN: %clang_cc1 %s -emit-llvm -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

__kernel void inc(__global int *x) { *x = *x + 1; }

__kernel void add(__global int *a, __global int *b, __global int *out) {
  *out = *a + *b;
}

void plain(int x) {}

// CIR: cir.func{{.*}} @inc{{.*}} cc(spir_kernel)
// CIR: cir.func{{.*}} @add{{.*}} cc(spir_kernel)
// CIR: cir.func{{.*}} @plain
// CIR-NOT: cc(spir_kernel)

// LLVM: define spir_kernel void @inc
// LLVM: define spir_kernel void @add

// OGCG: define spir_kernel void @inc
// OGCG: define spir_kernel void @add
