// clang/test/CodeGenOpenCL/coop-mat-codegen.cl
//
// Patch 5: CodeGen -- TargetExtType lowering, builtin IR emission,
//          SPIR-V intrinsic names.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_ext_kernel_cooperative_matrix \
// RUN:   -finclude-default-header -emit-llvm -O0 -o - %s \
// RUN:   | FileCheck %s

#define SCOPE     CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_t;

// ---------------------------------------------------------------------------
// 5a. coop_mat_load -> __spirv_CooperativeMatrixLoadKHR
//     Also verifies CooperativeMatrixType lowers to spirv.CooperativeMatrixKHR
//     TargetExtType (visible in the call signature).
// ---------------------------------------------------------------------------
kernel void test_load(__global float *ptr) {
    MatA_t a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_load
// CHECK: call
// CHECK-SAME: target("spirv.CooperativeMatrixKHR"
// CHECK-SAME: @__spirv_CooperativeMatrixLoadKHR

// ---------------------------------------------------------------------------
// 5b. coop_mat_store -> __spirv_CooperativeMatrixStoreKHR
// ---------------------------------------------------------------------------
kernel void test_store(__global float *ptr, MatA_t a) {
    coop_mat_store(ptr, a, ROW_MAJOR, 16);
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_store
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR

// ---------------------------------------------------------------------------
// 5c. coop_mat_mulAdd -> __spirv_CooperativeMatrixMulAddKHR
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 5d. Binary add (float element) -> __spirv_CooperativeMatrixFAdd
// ---------------------------------------------------------------------------
kernel void test_binary_add(__global float *ptr) {
    MatA_t a; MatA_t b; MatA_t r;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = a + b;
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_binary_add
// CHECK: call {{.*}} @__spirv_CooperativeMatrixFAdd

// ---------------------------------------------------------------------------
// 5e. Scalar multiply -> __spirv_CooperativeMatrixScalarMulKHR
// ---------------------------------------------------------------------------
kernel void test_scalar_mul(__global float *ptr, float s) {
    MatA_t a; MatA_t r;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = a * s;
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_scalar_mul
// CHECK: call {{.*}} @__spirv_CooperativeMatrixScalarMulKHR

// ---------------------------------------------------------------------------
// 5f. Unary minus -> __spirv_CooperativeMatrixScalarNeg
// ---------------------------------------------------------------------------
kernel void test_unary_neg(__global float *ptr) {
    MatA_t a; MatA_t r;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    r = -a;
    (void)r;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_unary_neg
// CHECK: call {{.*}} @__spirv_CooperativeMatrixScalarNeg
