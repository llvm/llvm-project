// RUN: %clang_cc1 -no-opaque-pointers %s -triple nvptx-unknown-unknown -emit-llvm -O0 -o - | FileCheck %s

void device_function() {
}
// CHECK-LABEL: define{{.*}} void @device_function()

__kernel void kernel_function() {
  device_function();
}
// CHECK-LABEL: define{{.*}} spir_kernel void @kernel_function()
// CHECK: call void @device_function()
// CHECK: !{{[0-9]+}} = !{void ()* @kernel_function, !"kernel", i32 1}

