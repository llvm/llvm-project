// clang/test/SemaOpenCL/coop-mat-sema-neg.cl
//
// Patch 4: Sema — negative tests for diagnostic paths.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

#define SCOPE     CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef int   __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_int_t;

// ---------------------------------------------------------------------------
// 1. Invalid scope value (0 is not CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP)
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(0, 16, 16, USE_A))) MatBadScope; // expected-error {{invalid argument of cooperative matrix attribute}}

// ---------------------------------------------------------------------------
// 2. Invalid use value (99 is not 0/1/2)
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(SCOPE, 16, 16, 99))) MatBadUse;  // expected-error {{invalid argument of cooperative matrix attribute}}

// ---------------------------------------------------------------------------
// 3. Mismatched element types in coop_mat_mulAdd
//    a/b are float matrices, c is an int matrix — should fire element type
//    mismatch diagnostic.
//    Use the two-step declare-then-assign pattern so that the coop_mat_load
//    fixup path fires correctly and we only get the intended mulAdd error.
// ---------------------------------------------------------------------------
kernel void test_muladd_type_mismatch(__global float *fptr,
                                      __global int   *iptr) {
    MatA_t     a;
    MatB_t     b;
    MatC_int_t c;
    a = coop_mat_load(fptr, ROW_MAJOR, 16);
    b = coop_mat_load(fptr, ROW_MAJOR, 16);
    c = coop_mat_load(iptr, ROW_MAJOR, 16);

    MatC_int_t result;
    result = coop_mat_mulAdd(a, b, c); // expected-error {{inconsistent cooperative matrix element type}}
    (void)result;
}

// ---------------------------------------------------------------------------
// 4. Assignment of coop_mat_load result to a plain scalar — must fire the
//    "should be assigned to cooperative matrix type variable" diagnostic.
//    This intentionally uses single-step initialization because the error
//    fires precisely when the LHS is NOT a coop mat type.
// ---------------------------------------------------------------------------
kernel void test_bad_assignment(__global float *ptr) {
    float bad;
	bad = coop_mat_load(ptr, ROW_MAJOR, 16); // expected-error {{builtin return value should be assigned to cooperative matrix type variable}}
    (void)bad;
}
