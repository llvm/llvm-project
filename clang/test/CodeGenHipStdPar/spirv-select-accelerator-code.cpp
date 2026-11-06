// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip --hipstdpar -fcuda-is-device \
// RUN:   -disable-llvm-optzns -emit-llvm -o - %s \
// RUN:   | FileCheck --implicit-check-not='@foo(' %s

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip --hipstdpar -fcuda-is-device \
// RUN:   -fgpu-rdc -disable-llvm-optzns -emit-llvm -o - %s \
// RUN:   | FileCheck --check-prefix=RDC %s

#define __global__ __attribute__((global))

// RDC: define {{.*}} @foo(
extern "C" void foo() {}

// CHECK: define {{.*}} @bar(
extern "C" void bar() {}

// CHECK: define spir_kernel void @kern(
extern "C" __global__ void kern() { bar(); }
