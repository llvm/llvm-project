// Test that using -emit-cir when clang is not built with CIR support gives a proper error message
// instead of crashing.

// This test should only run when CIR support is NOT enabled
// UNSUPPORTED: cir-support

// RUN: not %clang_cc1 -emit-cir %s 2>&1 | FileCheck %s
// CHECK: error: clang IR support not available, rebuild clang with -DCLANG_ENABLE_CIR=ON

int main(void) {
  return 0;
}
