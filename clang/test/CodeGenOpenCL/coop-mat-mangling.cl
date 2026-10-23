// Tests for function-name mangling of SPIR-V friendly cooperative matrix intrinsics.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 \
// RUN:   -o  - %s | FileCheck %s

#define SCOPE     memory_scope_sub_group
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

// --- float 16x16 (MulAdd group 1) ---
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatF_16x16_A;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatF_16x16_B;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatF_16x16_C;

// --- float 8x32 / 32x8 / 8x8 (MulAdd group 2 -- same elem type, diff dims) ---
typedef float __attribute__((coop_mat(SCOPE,  8, 32, USE_A))) MatF_8x32_A;
typedef float __attribute__((coop_mat(SCOPE, 32,  8, USE_B))) MatF_32x8_B;
typedef float __attribute__((coop_mat(SCOPE,  8,  8, USE_C))) MatF_8x8_C;

// --- int 16x16 (MulAdd group 3 -- different element type to group 1) ---
typedef int __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatI_16x16_A;
typedef int __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatI_16x16_B;
typedef int __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatI_16x16_C;

// MulAdd test
kernel void test_muladd_f32_16x16(__global float *ptr) {
    MatF_16x16_A a;
    MatF_16x16_B b;
    MatF_16x16_C c, r;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, ROW_MAJOR, 16);
    c = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = coop_mat_mulAdd(a, b, c);
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_muladd_f32_16x16
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR_f32_sc3_16x16_u0_f32_sc3_16x16_u1_f32_sc3_16x16_u2

// Same element type as test_muladd_f32_16x16 (float), but different
// matrix dimensions
// 16x16 x 16x16 -> 16x16   vs   8x32 x 32x8 -> 8x8
// Must produce a distinct mangled function name.
kernel void test_muladd_f32_8x32(__global float *ptr) {
    MatF_8x32_A  a;
    MatF_32x8_B  b;
    MatF_8x8_C   c, r;
    a = coop_mat_load(ptr, ROW_MAJOR, 8);
    b = coop_mat_load(ptr, ROW_MAJOR, 32);
    c = coop_mat_load(ptr, ROW_MAJOR, 8);
    r = coop_mat_mulAdd(a, b, c);
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_muladd_f32_8x32
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR_f32_sc3_8x32_u0_f32_sc3_32x8_u1_f32_sc3_8x8_u2

// Same dimensions as test_muladd_f32_16x16 (16x16), but different element
// types.
// float vs int
// Must produce a distinct mangled function name.
kernel void test_muladd_i32_16x16(__global int *iptr) {
    MatI_16x16_A a;
    MatI_16x16_B b;
    MatI_16x16_C c, r;
    a = coop_mat_load(iptr, ROW_MAJOR, 16);
    b = coop_mat_load(iptr, ROW_MAJOR, 16);
    c = coop_mat_load(iptr, ROW_MAJOR, 16);
    r = coop_mat_mulAdd(a, b, c);
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_muladd_i32_16x16
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR_i32_sc3_16x16_u0_i32_sc3_16x16_u1_i32_sc3_16x16_u2

// Load: different pointer address spaces
// __global (addrspace 1) vs __local (addrspace 3)
// Must produce distinct mangled function names.
kernel void test_load_global(__global float *ptr) {
    MatF_16x16_A a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a;
}
kernel void test_load_local(__local float *ptr) {
    MatF_16x16_A a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_load_global
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_16x16_u0
// CHECK-SAME: ptr addrspace(1)
// CHECK-LABEL: @__clang_ocl_kern_imp_test_load_local
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLoadKHR_local_f32_sc3_16x16_u0
// CHECK-SAME: ptr addrspace(3)

// Load: different pointer address spaces
// __global (addrspace 1) vs __local (addrspace 3)
// Must produce distinct mangled function names.
kernel void test_store_global(__global float *ptr) {
    MatF_16x16_A a;
    a = coop_mat_init(0.0f);
    coop_mat_store(ptr, a, ROW_MAJOR, 16);
    (void)a;
}
kernel void test_store_local(__local float *ptr) {
    MatF_16x16_A a;
    a = coop_mat_init(0.0f);
    coop_mat_store(ptr, a, ROW_MAJOR, 16);
    (void)a;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_store_global
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR_global_f32_sc3_16x16_u0
// CHECK-SAME: ptr addrspace(1)
// CHECK-LABEL: @__clang_ocl_kern_imp_test_store_local
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR_local_f32_sc3_16x16_u0
// CHECK-SAME: ptr addrspace(3)
