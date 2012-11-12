// RUN: %clang_cc1 %s -ffake-address-space-map -emit-llvm -o - | FileCheck %s

__kernel void foo(void) {
  // CHECK: @foo.i = internal addrspace(2)
  __local int i;
  ++i;
}
