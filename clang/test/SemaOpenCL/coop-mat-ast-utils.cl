// Test structural equivalence (no diagnostics on compatible pair)
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics

#define SCOPE memory_scope_sub_group
#define USE_A CLK_COOPERATIVE_MATRIX_A
#define USE_B CLK_COOPERATIVE_MATRIX_B
#define USE_C CLK_COOPERATIVE_MATRIX_ACCUMULATOR

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_t;

// Structural equivalence — produce no diagnostic.
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_alias;
void test_structural_equiv(void) {
    MatA_t    *p = 0;
    MatA_alias *q = p;  // same canonical type
    (void)q;
}
