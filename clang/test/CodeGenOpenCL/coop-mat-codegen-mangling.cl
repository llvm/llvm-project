// clang/test/CodeGenOpenCL/coop-mat-codegen-mangling.cl
//
// Tests for function-name mangling of SPIR-V cooperative matrix intrinsics.
// Covers three mangling axes:
//   (A) coop_mat_mulAdd  -- same element type, different matrix dimensions / scope / use
//   (B) coop_mat_mulAdd  -- different element types (float vs int)
//   (C) coop_mat_load    -- different pointer address spaces
//   (D) coop_mat_store   -- different pointer address spaces
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -emit-llvm -O0 -o %t.ll %s
// RUN: FileCheck --check-prefix=CHECK      %s < %t.ll
// RUN: FileCheck --check-prefix=CHECK-DECL %s < %t.ll

#define SCOPE     CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

// ---------------------------------------------------------------------------
// Matrix type aliases used across tests
// ---------------------------------------------------------------------------

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

// ===========================================================================
// (A) MulAdd: same element type (float), different matrix dimensions
//     16x16 x 16x16 -> 16x16   vs   8x32 x 32x8 -> 8x8
//     Must produce TWO distinct __spirv_CooperativeMatrixMulAddKHR_* symbols.
// ===========================================================================

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
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR
// CHECK-SAME: _f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}
// CHECK-SAME: _f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}
// CHECK-SAME: _f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}

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
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR
// CHECK-SAME: _f32_sc{{[0-9]+}}_8x32_u{{[0-9]+}}
// CHECK-SAME: _f32_sc{{[0-9]+}}_32x8_u{{[0-9]+}}
// CHECK-SAME: _f32_sc{{[0-9]+}}_8x8_u{{[0-9]+}}

// ===========================================================================
// (B) MulAdd: different element types -- float 16x16 vs int 16x16
//     Same dimensions, different elem type -> must produce two distinct symbols.
// ===========================================================================

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
// CHECK: call {{.*}} @__spirv_CooperativeMatrixMulAddKHR
// CHECK-SAME: _i32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}

// ===========================================================================
// (C) Load: different pointer address spaces
//     __global (addrspace 1) vs __local (addrspace 3) vs private (addrspace 0)
//     Must produce THREE distinct __spirv_CooperativeMatrixLoadKHR_* symbols.
// ===========================================================================

kernel void test_load_global(__global float *ptr) {
    MatF_16x16_A a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_load_global
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}
// CHECK-SAME: ptr addrspace(1)

kernel void test_load_local(__local float *ptr) {
    MatF_16x16_A a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_load_local
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLoadKHR_local_f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}
// CHECK-SAME: ptr addrspace(3)

// ===========================================================================
// (D) Store: different pointer address spaces
//     Same matrix type, __global vs __local -> two distinct symbols.
// ===========================================================================

kernel void test_store_global(__global float *ptr, MatF_16x16_A a) {
    coop_mat_store(ptr, a, ROW_MAJOR, 16);
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_store_global
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR_global_f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}
// CHECK-SAME: ptr addrspace(1)

kernel void test_store_local(__local float *ptr, MatF_16x16_A a) {
    coop_mat_store(ptr, a, ROW_MAJOR, 16);
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_store_local
// CHECK: call {{.*}} @__spirv_CooperativeMatrixStoreKHR_local_f32_sc{{[0-9]+}}_16x16_u{{[0-9]+}}
// CHECK-SAME: ptr addrspace(3)

// ===========================================================================
// (E) Negative: confirm NO cross-contamination between address space variants
//     i.e. the _global load function is NOT called for the _local case.
// ===========================================================================

kernel void test_no_cross_contamination(__global float *gptr,
                                        __local  float *lptr) {
    MatF_16x16_A ag, al;
    ag = coop_mat_load(gptr, ROW_MAJOR, 16);
    al = coop_mat_load(lptr, ROW_MAJOR, 16);
    (void)ag; (void)al;
}
// CHECK-LABEL: @__clang_ocl_kern_imp_test_no_cross_contamination
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32
// CHECK: call {{.*}} @__spirv_CooperativeMatrixLoadKHR_local_f32
// CHECK-NOT: @__spirv_CooperativeMatrixLoadKHR_global_f32_{{.*}}addrspace(3)
// CHECK-NOT: @__spirv_CooperativeMatrixLoadKHR_local_f32_{{.*}}addrspace(1)

// Verify distinct declare lines exist (these appear at end of IR).
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_16x16_u0
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_16x16_u1
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_16x16_u2
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixMulAddKHR_f32_sc3_16x16_u0_f32_sc3_16x16_u1_f32_sc3_16x16_u2
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_8x32_u0
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_32x8_u1
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_f32_sc3_8x8_u2
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixMulAddKHR_f32_sc3_8x32_u0_f32_sc3_32x8_u1_f32_sc3_8x8_u2
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_i32_sc3_16x16_u0
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_i32_sc3_16x16_u1
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_global_i32_sc3_16x16_u2
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixMulAddKHR_i32_sc3_16x16_u0_i32_sc3_16x16_u1_i32_sc3_16x16_u2
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixLoadKHR_local_f32_sc3_16x16_u0
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixStoreKHR_global_f32_sc3_16x16_u0
// CHECK-DECL: declare {{.*}} @__spirv_CooperativeMatrixStoreKHR_local_f32_sc3_16x16_u0
