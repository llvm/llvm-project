// RUN: %clang_cc1 -x cl -cl-opt-disable -cl-std=CL1.2 -emit-llvm -ffake-address-space-map %s -o - -verify | FileCheck %s
// expected-no-diagnostics

// CHECK: @foo = external addrspace(3) constant float
extern constant float foo;

kernel void test(global float* buf) {
  buf[0] += foo;
}
