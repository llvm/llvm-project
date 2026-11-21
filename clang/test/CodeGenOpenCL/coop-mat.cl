// Cooperative matrix tests for SPIR-V friendly LLVM IR code generation from
// clang frontend.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 \
// RUN:   -o - %s | FileCheck %s

#define SCOPE     memory_scope_sub_group
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_t;

// coop_mat_load -> __spirv_CooperativeMatrixLoadKHR
//     Also verifies CooperativeMatrixType lowers to spirv.CooperativeMatrixKHR
//     TargetExtType (visible in the call signature).
kernel void test_load(__global float *ptr, __global uchar4 *ptr4, __global char *ptrc) {
    MatA_t a_1;
    a_1 = coop_mat_load(ptr, a_1, ROW_MAJOR, 16);
    (void)a_1;
    MatA_t a_2 = coop_mat_load(ptr, a_2, ROW_MAJOR, 16);
    (void)a_2;
    MatA_t a_3;
    a_3 = (coop_mat_load(ptr, a_3, ROW_MAJOR, 16));
    (void)a_3;
    MatA_t a_4 = (coop_mat_load(ptr, a_4, ROW_MAJOR, 16));
    (void)a_4;
    MatA_t a_5 = coop_mat_load(ptr4, a_5, ROW_MAJOR, 16);
    (void)a_5;
    MatA_t a_6 = coop_mat_load(ptrc, a_6, ROW_MAJOR, 64);
    (void)a_6;
    int x = coop_mat_length(coop_mat_load(ptrc, a_6, ROW_MAJOR, 64));
    (void)x;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_load
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR
// CHECK: call
// CHECK-SAME: @__spirv_CooperativeMatrixLengthKHR

// coop_mat_store -> __spirv_CooperativeMatrixStoreKHR
kernel void test_store(__global float *ptr, __global float4 *ptr4) {
    MatA_t a_1;
    coop_mat_store(ptr, a_1, ROW_MAJOR, 16);
    (void)a_1;
    MatA_t a_2;
    coop_mat_store(ptr4, a_2, ROW_MAJOR, 16);
    (void)a_2;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_store
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR

// coop_mat_mulAdd -> __spirv_CooperativeMatrixMulAddKHR
kernel void test_muladd(__global float *ptr) {
    MatA_t a; MatB_t b; MatC_t c; MatC_t r;
    a = coop_mat_load(ptr, a, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, b, ROW_MAJOR, 16);
    c = coop_mat_load(ptr, c, ROW_MAJOR, 16);
    r = coop_mat_mulAdd(a, b, c);
    int x = coop_mat_length(coop_mat_mulAdd(a, b, c));
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_muladd
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLengthKHR

// Binary ops -> __spirv_FAdd etc
typedef int __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_int_t;
typedef unsigned int __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_uint_t;
kernel void test_binary_ops(__global float *ptr, __global int *ptr_i, __global unsigned int *ptr_u) {
    MatA_t a, b, r1, r2, r3, r4;
    a = coop_mat_load(ptr, a, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, b, ROW_MAJOR, 16);
    r1 = a + b;
    (void)r1;
    r2 = a - b;
    (void)r2;
    r3 = a * b;
    (void)r3;
    r4 = a / b;
    (void)r4;
    MatA_int_t ai, bi, r1i, r2i, r3i, r4i;
    ai = coop_mat_load(ptr_i, ai, ROW_MAJOR, 16);
    bi = coop_mat_load(ptr_i, bi, ROW_MAJOR, 16);
    r1i = ai + bi;
    (void)r1i;
    r2i = ai - bi;
    (void)r2i;
    r3i = ai * bi;
    (void)r3i;
    r4i = ai / bi;
    (void)r4i;
    MatA_uint_t au, bu, r1u, r2u, r3u, r4u;
    au = coop_mat_load(ptr_u, au, ROW_MAJOR, 16);
    bu = coop_mat_load(ptr_u, bu, ROW_MAJOR, 16);
    r1u = au + bu;
    (void)r1u;
    r2u = au - bu;
    (void)r2u;
    r3u = au * bu;
    (void)r3u;
    r4u = au / bu;
    (void)r4u;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_binary_ops
// CHECK: call {{.*}} @__spirv_FAdd
// CHECK: call {{.*}} @__spirv_FSub
// CHECK: call {{.*}} @__spirv_FMul
// CHECK: call {{.*}} @__spirv_FDiv
// CHECK: call {{.*}} @__spirv_IAdd
// CHECK: call {{.*}} @__spirv_ISub
// CHECK: call {{.*}} @__spirv_IMul
// CHECK: call {{.*}} @__spirv_SDiv
// CHECK: call {{.*}} @__spirv_IAdd
// CHECK: call {{.*}} @__spirv_ISub
// CHECK: call {{.*}} @__spirv_IMul
// CHECK: call {{.*}} @__spirv_UDiv

// Scalar multiply -> __spirv_CooperativeMatrixScalarMulKHR
kernel void test_scalar_mul(__global float *ptr, float s) {
    MatA_t a; MatA_t r;
    a = coop_mat_load(ptr, a, ROW_MAJOR, 16);
    r = a * s;
    (void)r;
    MatA_t r2;
    r2 = s * a;
    (void)r2;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_scalar_mul
// CHECK: call {{.*}} @__spirv_CooperativeMatrixScalarMulKHR
// CHECK: call {{.*}} @__spirv_CooperativeMatrixScalarMulKHR

// Unary minus -> __spirv_FNegate
kernel void test_unary_neg(__global float *ptr) {
    MatA_t a; MatA_t r;
    a = coop_mat_load(ptr, a, ROW_MAJOR, 16);
    r = -a;
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_unary_neg
// CHECK: call {{.*}} @__spirv_FNegate
