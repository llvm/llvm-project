// RUN: %clang_cc1 %s -cl-opt-disable -ffake-address-space-map -emit-llvm -o - | FileCheck %s

// CHECK: @array = addrspace({{[0-9]+}}) constant
__constant float array[2] = {0.0f, 1.0f};

kernel void test(global float *out) {
  *out = array[0];
}
