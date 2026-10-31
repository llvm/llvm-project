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
    a_1 = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a_1;
    MatA_t a_2 = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a_2;
    MatA_t a_3;
    a_3 = (coop_mat_load(ptr, ROW_MAJOR, 16));
    (void)a_3;
    MatA_t a_4 = (coop_mat_load(ptr, ROW_MAJOR, 16));
    (void)a_4;
    MatA_t a_5 = coop_mat_load(ptr4, ROW_MAJOR, 16);
    (void)a_5;
    MatA_t a_6 = coop_mat_load(ptrc, ROW_MAJOR, 64);
    (void)a_6;
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
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, ROW_MAJOR, 16);
    c = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = coop_mat_mulAdd(a, b, c);
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_muladd
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR

// Binary add (float element) -> __spirv_FAdd
kernel void test_binary_add(__global float *ptr) {
    MatA_t a; MatA_t b; MatA_t r;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = a + b;
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_binary_add
// CHECK: call {{.*}} @__spirv_FAdd

// Scalar multiply -> __spirv_CooperativeMatrixScalarMulKHR
kernel void test_scalar_mul(__global float *ptr, float s) {
    MatA_t a; MatA_t r;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
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
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = -a;
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_unary_neg
// CHECK: call {{.*}} @__spirv_FNegate
