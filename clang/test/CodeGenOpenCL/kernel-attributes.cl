// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

kernel  __attribute__((vec_type_hint(int))) __attribute__((reqd_work_group_size(1,2,4))) void kernel1(int a) {}
// CHECK: define spir_kernel void @kernel1(i32 {{[^%]*}}%a) {{[^{]+}} !vec_type_hint ![[MD1:[0-9]+]] !reqd_work_group_size ![[MD2:[0-9]+]]

kernel __attribute__((vec_type_hint(uint4))) __attribute__((work_group_size_hint(8,16,32))) void kernel2(int a) {}
// CHECK: define spir_kernel void @kernel2(i32 {{[^%]*}}%a) {{[^{]+}} !vec_type_hint ![[MD3:[0-9]+]] !work_group_size_hint ![[MD4:[0-9]+]]

kernel __attribute__((intel_reqd_sub_group_size(8))) void kernel3(int a) {}
// CHECK: define spir_kernel void @kernel3(i32 {{[^%]*}}%a) {{[^{]+}} !intel_reqd_sub_group_size ![[MD5:[0-9]+]]

// CHECK: [[MD1]] = !{i32 undef, i32 1}
// CHECK: [[MD2]] = !{i32 1, i32 2, i32 4}
// CHECK: [[MD3]] = !{<4 x i32> undef, i32 0}
// CHECK: [[MD4]] = !{i32 8, i32 16, i32 32}
// CHECK: [[MD5]] = !{i32 8}
