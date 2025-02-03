// RUN: %clang_cc1 %s -cl-std=CL1.2 -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s

// Test that pointer arguments to kernels are assumed to be ABI aligned.

struct __attribute__((packed, aligned(1))) packed {
  int i32;
};

typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(2))) float float2;

kernel void test(
    global int *i32,
    global long *i64,
    global int4 *v4i32,
    global float2 *v2f32,
    global void *v,
    global struct packed *p) {
// CHECK-LABEL: spir_kernel void @test(
// CHECK-SAME: ptr noundef readnone align 4 captures(none) %i32,
// CHECK-SAME: ptr noundef readnone align 8 captures(none) %i64,
// CHECK-SAME: ptr noundef readnone align 16 captures(none) %v4i32,
// CHECK-SAME: ptr noundef readnone align 8 captures(none) %v2f32,
// CHECK-SAME: ptr noundef readnone captures(none) %v,
// CHECK-SAME: ptr noundef readnone align 1 captures(none) %p)
}
