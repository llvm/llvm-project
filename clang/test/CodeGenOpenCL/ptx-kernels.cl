// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -emit-llvm -o - | FileCheck %s

void device_function() {
}
// CHECK: define ptx_device void @device_function()

__kernel void kernel_function() {
}
// CHECK: define ptx_kernel void @kernel_function()

