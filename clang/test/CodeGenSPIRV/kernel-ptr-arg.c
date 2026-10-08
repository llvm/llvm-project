// RUN: %clang_cc1 -triple spirv64 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv32 %s -emit-llvm -o - | FileCheck %s

// CHECK: define spir_func void @func(ptr noundef %{{.*}})
void func(int* arg) {
}

// CHECK: define spir_kernel void @kernel(ptr addrspace(1) noundef %{{.*}})
void __attribute__((device_kernel)) kernel(int* arg) {
// CHECK: call spir_func{{.*}} void @func(ptr noundef %{{.*}})
  func(arg);
}

// CHECK: define spir_kernel void @kernel_spec(ptr addrspace(2) noundef %{{.*}})
void __attribute__((device_kernel)) kernel_spec(__attribute__((address_space(2))) int* arg) {
// CHECK: call spir_func{{.*}} void @func(ptr noundef %{{.*}})
  func((int*)arg);
}
