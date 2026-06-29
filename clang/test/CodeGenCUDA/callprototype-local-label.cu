// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_75 \
// RUN:   -fcuda-is-device -S -o - -x cuda %s \
// RUN:   | FileCheck %s

// Test that a global named 'prototype_0' does not cause a callprototype label
// collision: the label must be '$L__prototype_0', not 'prototype_0'.
// extern "C" is used to keep the PTX global name unmangled.

#define __device__ __attribute__((device))

extern "C" {

__device__ int simple_func() { return 42; }

__device__ int (*prototype_0)(int, int, int, int) = nullptr;
__device__ int call_via_prototype_0(int a, int b, int c, int d) {
    if (prototype_0 != nullptr)
        return prototype_0(a, b, c, d);
    return a + b + c + d;
}

} // extern "C"

// CHECK: .visible .global .align 8 .u64 prototype_0;
// CHECK-LABEL: .visible .func  (.param .b32 func_retval0) call_via_prototype_0(
// CHECK: $L__prototype_0 : .callprototype (.param .b32 _) _ (.param .b32 _, .param .b32 _, .param .b32 _, .param .b32 _);
// CHECK: call (retval0), %rd{{[0-9]+}}, (param0, param1, param2, param3), $L__prototype_0;
// CHECK-NOT: prototype_0 : .callprototype
