// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -emit-llvm -O0 -o - | FileCheck %s

void device_function() {
}
// CHECK-LABEL: define{{.*}} void @device_function()

__kernel void kernel_function() {
  device_function();
}
// CHECK-LABEL: define{{.*}} ptx_kernel void @kernel_function()
// CHECK: call void @device_function()
