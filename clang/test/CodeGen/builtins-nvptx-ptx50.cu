// RUN: %clang_cc1 -no-opaque-pointers -triple nvptx64-unknown-unknown -target-cpu sm_60 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK %s
//
// RUN: %clang_cc1 -no-opaque-pointers -triple nvptx-unknown-unknown -target-cpu sm_50 \
// RUN:   -fcuda-is-device -S -o /dev/null -x cuda -verify %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

// We have to keep all builtins that depend on particular target feature in the
// same function, because the codegen will stop after the very first function
// that encounters an error, so -verify will not be able to find errors in
// subsequent functions.

// CHECK-LABEL: test_fn
__device__ void test_fn(double d, double* double_ptr) {
  // CHECK: atomicrmw fadd double* {{.*}} seq_cst, align 8
  // expected-error@+1 {{'__nvvm_atom_add_gen_d' needs target feature sm_60}}
  __nvvm_atom_add_gen_d(double_ptr, d);
}
