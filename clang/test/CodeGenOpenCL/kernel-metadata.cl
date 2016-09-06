// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void normal_function() {
}

__kernel void kernel_function() {
}

// CHECK: define void @kernel_function() {{[^{]+}} !kernel_arg_addr_space ![[MD:[0-9]+]] !kernel_arg_access_qual ![[MD]] !kernel_arg_type ![[MD]] !kernel_arg_base_type ![[MD]] !kernel_arg_type_qual ![[MD]] {
// CHECK: ![[MD]] = !{}
