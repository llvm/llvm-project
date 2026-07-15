// Tests for coop_mat_init and coop_mat_length — Sema (positive).
//
// coop_mat_init and coop_mat_length have no entries in CheckBuiltinFunctionCall,
// so arg-count and arg-type validation are not performed by Sema for these two
// builtins. Only the positive (valid-code) path and the one assignment-target
// diagnostic (err_coop_matrix_assignment, checked via AddInitializerToDecl) are
// tested here.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_ext_kernel_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

// expected-no-diagnostics (sections 1-6 must compile cleanly)

// ---------------------------------------------------------------------------
// Type definitions shared across all tests.
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_A)))           MatA_t;
typedef float __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_B)))           MatB_t;
typedef float __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_ACCUMULATOR))) MatC_t;
typedef int   __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_A)))           MatA_int_t;

// ---------------------------------------------------------------------------
// 1. coop_mat_init: initialise each use role with a float scalar.
//    Two-step (declare then assign) is required because the return type is
//    fixed up in AddInitializerToDecl / CreateBuiltinBinOp.
// ---------------------------------------------------------------------------
void test_coop_mat_init_basic(void) {
    MatA_t a;
    MatB_t b;
    MatC_t c;
    a = coop_mat_init(1.0f);
    b = coop_mat_init(2.0f);
    c = coop_mat_init(0.0f);
}

// ---------------------------------------------------------------------------
// 2. coop_mat_init: integer element type.
// ---------------------------------------------------------------------------
void test_coop_mat_init_int(void) {
    MatA_int_t m;
    m = coop_mat_init(42);
}

// ---------------------------------------------------------------------------
// 3. coop_mat_init: re-initialise the same variable (chained assigns).
// ---------------------------------------------------------------------------
void test_coop_mat_init_chained(void) {
    MatC_t acc;
    acc = coop_mat_init(0.0f);
    acc = coop_mat_init(1.0f);
}

// ---------------------------------------------------------------------------
// 4. coop_mat_length: basic usage — returns unsigned int directly.
//    No two-step required for the length call itself.
// ---------------------------------------------------------------------------
void test_coop_mat_length_basic(void) {
    MatA_t a;
    a = coop_mat_init(0.0f);
    unsigned int len = coop_mat_length(a);
    (void)len;
}

// ---------------------------------------------------------------------------
// 5. coop_mat_length: all three use roles.
// ---------------------------------------------------------------------------
void test_coop_mat_length_roles(void) {
    MatA_t a;   a = coop_mat_init(0.0f);
    MatB_t b;   b = coop_mat_init(0.0f);
    MatC_t c;   c = coop_mat_init(0.0f);
    unsigned int la = coop_mat_length(a);
    unsigned int lb = coop_mat_length(b);
    unsigned int lc = coop_mat_length(c);
    (void)la; (void)lb; (void)lc;
}

// ---------------------------------------------------------------------------
// 6. coop_mat_length: result used in arithmetic.
// ---------------------------------------------------------------------------
void test_coop_mat_length_arith(void) {
    MatA_t a;
    a = coop_mat_init(0.0f);
    unsigned int half_len = coop_mat_length(a) / 2u;
    (void)half_len;
}
