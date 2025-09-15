// RUN: %clang_cc1 -triple spirv64-intel %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-WITH-64 %s
// RUN: %clang_cc1 -triple spirv32-intel %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-WITH-32 %s
// RUN: %clang_cc1 -triple spir-intel %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-WITHOUT %s
// RUN: %clang_cc1 -triple spir64-intel %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-WITHOUT %s

// CHECK-WITH-64: spir_func void @foo(ptr addrspace(4) noundef %param) addrspace(9) #0 {
// CHECK-WITH-32: spir_func void @foo(ptr addrspace(4) noundef %param) #0 {

// CHECK-WITHOUT: spir_func void @foo(ptr noundef %param) #0 {
void foo(int *param) {
}
