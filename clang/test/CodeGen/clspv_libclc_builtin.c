// RUN: %clang_cc1 -triple spir -emit-llvm %s -o - | FileCheck %s

// CHECK: @foo()
// CHECK-SAME: !clspv_libclc_builtin

void __attribute__((clspv_libclc_builtin)) foo() {}
