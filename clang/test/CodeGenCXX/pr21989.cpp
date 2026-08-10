// REQUIRES: asserts
// RUN: not %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s 2>&1 | FileCheck %s

struct {
  void __attribute__((used)) f() {}
};
// CHECK: 2 errors generated.

// Emit the errors, but don't assert.
