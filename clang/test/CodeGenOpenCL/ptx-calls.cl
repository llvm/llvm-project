// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -emit-llvm -O0 -o - | FileCheck %s

void device_function() {
}
// CHECK: define ptx_device void @device_function()

__kernel void kernel_function() {
  device_function();
}
// CHECK: define ptx_kernel void @kernel_function()
// CHECK: call ptx_device void @device_function()

