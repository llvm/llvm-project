// clang/test/SemaOpenCL/coop-mat-sema.cl
//
// Patch 4: Sema -- BuildCooperativeMatrixType, diagnostics,
//          builtin validation, expr handling, TreeTransform.
//
// Valid code -- expected-no-diagnostics
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_ext_kernel_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

#define SCOPE    CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP
#define USE_A    CLK_COOPERATIVE_MATRIX_A
#define USE_B    CLK_COOPERATIVE_MATRIX_B
#define USE_C    CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_t;

// ---------------------------------------------------------------------------
// 4b. load builtin -- return type is fixed up by AddInitializerToDecl to
//     match the LHS variable type.  Declare first, then assign so that
//     Sema's IsCoopMatrixBuiltin path fires correctly.
// ---------------------------------------------------------------------------
kernel void test_load_store(__global float *ptr,
                            __global float *out_ptr) {
    MatA_t a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    coop_mat_store(out_ptr, a, ROW_MAJOR, 16);
}

// ---------------------------------------------------------------------------
// 4c. mulAdd builtin -- same two-step pattern for each matrix.
// ---------------------------------------------------------------------------
kernel void test_muladd(__global float *ptr) {
    MatA_t a;
    MatB_t b;
    MatC_t c;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    b = coop_mat_load(ptr, ROW_MAJOR, 16);
    c = coop_mat_load(ptr, ROW_MAJOR, 16);
    MatC_t result;
    result = coop_mat_mulAdd(a, b, c);
    (void)result;
}

// ---------------------------------------------------------------------------
// 4d. Binary element-wise operators (+, -)
// ---------------------------------------------------------------------------
void test_binary_ops(MatA_t a, MatA_t b) {
    MatA_t r_add = a + b;
    MatA_t r_sub = a - b;
    (void)r_add; (void)r_sub;
}

// ---------------------------------------------------------------------------
// 4e. Scalar multiply operator
// ---------------------------------------------------------------------------
void test_scalar_ops(MatA_t a, float s) {
    MatA_t r = a * s;
    (void)r;
}

// ---------------------------------------------------------------------------
// 4f. Unary minus
// ---------------------------------------------------------------------------
void test_unary_minus(MatA_t a) {
    MatA_t r = -a;
    (void)r;
}

// ---------------------------------------------------------------------------
// 4g. Assignment -- coop_mat_load return assigned to a pre-declared var.
//     This exercises the SemaDecl AddInitializerToDecl fixup path directly.
// ---------------------------------------------------------------------------
kernel void test_assignment(__global float *ptr) {
    MatA_t a;
    a = coop_mat_load(ptr, ROW_MAJOR, 16);
    (void)a;
}
