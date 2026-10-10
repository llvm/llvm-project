// Sema — negative tests for diagnostic paths.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fsyntax-only -verify %s

#define SCOPE     memory_scope_sub_group
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

// ---------------------------------------------------------------------------
// Use of cooperative matrix type without specifying pragma
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatX_t; // expected-error {{cooperative matrix types require OpenCL extension 'cl_khr_cooperative_matrix' to be enabled via pragma}}

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t; // expected-note {{declared here}}
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef int   __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_int_t;

// ---------------------------------------------------------------------------
// Invalid scope value (0 is not memory_scope_sub_group)
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(0, 16, 16, USE_A))) MatBadScope; // expected-error {{invalid 'scope' argument of cooperative matrix attribute}}

// ---------------------------------------------------------------------------
// Invalid use value (99 is not 0/1/2)
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(SCOPE, 16, 16, 99))) MatBadUse;  // expected-error {{invalid 'use' argument of cooperative matrix attribute}}

// ---------------------------------------------------------------------------
// Mismatched element types in coop_mat_mulAdd
//    a/b are float matrices, c is an int matrix — should fire element type
//    mismatch diagnostic.
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
    result = coop_mat_mulAdd(a, b, c); // expected-error {{inconsistent cooperative matrix element types}}
    (void)result;
}

// ---------------------------------------------------------------------------
// Assignment of coop_mat_load result to a plain scalar — must fire the
// "should be assigned to cooperative matrix type variable" diagnostic.
// ---------------------------------------------------------------------------
kernel void test_bad_assignment(__global float *ptr) {
    float bad;
	bad = coop_mat_load(ptr, ROW_MAJOR, 16); // expected-error {{builtin return value should be assigned to cooperative matrix type variable}}
    (void)bad;
}

// ---------------------------------------------------------------------------
// Kernel parameter cannot be of cooperative matrix type.
// ---------------------------------------------------------------------------
kernel void test_bad_kernel_param(MatA_t a) {} // expected-error {{cannot be used as the type of a kernel parameter}}

// ---------------------------------------------------------------------------
// Invalid memory layout for cooperative matrix load.
// ---------------------------------------------------------------------------
kernel void test_invalid_load_layout(__global float *ptr) {
    MatA_t a;
    a = coop_mat_load(ptr, 99, 16); // expected-error {{memory layout for cooperative matrix load/store should be CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR or CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR}}
}

// ---------------------------------------------------------------------------
// Invalid memory layout for cooperative matrix store.
// ---------------------------------------------------------------------------
kernel void test_invalid_store_layout(__global float *ptr) {
    MatA_t a;
    coop_mat_store(ptr, a, 99, 16); // expected-error {{memory layout for cooperative matrix load/store should be CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR or CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR}}
}

// ---------------------------------------------------------------------------
// Invalid stride type for cooperative matrix load.
// The stride must have type size_t.
// ---------------------------------------------------------------------------
kernel void test_invalid_load_stride(__global float *ptr) {
    MatA_t a;
    int stride = 16;
    a = coop_mat_load(ptr, ROW_MAJOR, stride); // expected-error {{stride argument for load/store of cooperative matrix must have type 'size_t'}}
}

// ---------------------------------------------------------------------------
// Invalid stride type for cooperative matrix store.
// ---------------------------------------------------------------------------
kernel void test_invalid_store_stride(__global float *ptr) {
    MatA_t a;
    int stride = 16;
    coop_mat_store(ptr, a, ROW_MAJOR, stride); // expected-error {{stride argument for load/store of cooperative matrix must have type 'size_t'}}
}

// ---------------------------------------------------------------------------
// Cooperative matrix element type must match the pointed-to buffer type.
// ---------------------------------------------------------------------------
kernel void test_load_element_pointer_mismatch(__global int *iptr) {
    MatA_t a;
    a = coop_mat_load(iptr, ROW_MAJOR, 16); // expected-error {{inconsistent between cooperative matrix element type and buffer pointer type}}
}

// ---------------------------------------------------------------------------
// The fourth coop_mat_mulAdd argument must have the cooperative-matrix
// operands enum type.
// ---------------------------------------------------------------------------
kernel void test_invalid_muladd_memory_operand_type() {
    MatA_t a;
    MatB_t b;
    MatC_int_t c;

    // The literal has type float rather than the required enum type.
    (void)coop_mat_mulAdd(a, b, c, 16.2f); // expected-error {{fourth argument of 'coop_mat_mulAdd' must have enum type}}
}

// ---------------------------------------------------------------------------
// coop_mat_mulAdd arguments must be cooperative matrix types.
// ---------------------------------------------------------------------------
kernel void test_invalid_muladd_argument(__global float *ptr) {
    MatA_t a;
    MatB_t b;
    MatC_int_t c;

    (void)coop_mat_mulAdd(a, b, 1.0f); // expected-error {{argument must be a valid cooperative matrix type}}
}

// ---------------------------------------------------------------------------
// coop_mat_mulAdd requires the first operand to have use A.
// ---------------------------------------------------------------------------
kernel void test_invalid_muladd_a_use(__global float *ptr) {
    MatB_t b;
    MatB_t b2;
    MatC_int_t c;

    (void)coop_mat_mulAdd(b, b2, c); // expected-error {{argument of cooperative matrix must have 'CLK_COOPERATIVE_MATRIX_A' use}}
}


// ---------------------------------------------------------------------------
// coop_mat_mulAdd requires the second operand to have use B.
// ---------------------------------------------------------------------------
kernel void test_invalid_muladd_b_use(__global float *ptr) {
    MatA_t a;
    MatA_t a2;
    MatC_int_t c;

    (void)coop_mat_mulAdd(a, a2, c); // expected-error {{argument of cooperative matrix must have 'CLK_COOPERATIVE_MATRIX_B' use}}
}

// ---------------------------------------------------------------------------
// coop_mat_mulAdd requires the third operand to have accumulator use.
// ---------------------------------------------------------------------------
kernel void test_invalid_muladd_accumulator_use(__global float *ptr) {
    MatA_t a;
    MatB_t b;
    MatA_t c;

    (void)coop_mat_mulAdd(a, b, c); // expected-error {{argument of cooperative matrix must have 'CLK_COOPERATIVE_MATRIX_ACCUMULATOR' use}}
}

// ---------------------------------------------------------------------------
// Invalid cooperative matrix element type.
// ---------------------------------------------------------------------------
typedef bool __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatBadElement_t; // expected-error {{'bool' is invalid cooperative matrix element type}}

// ---------------------------------------------------------------------------
// Cooperative matrices with incompatible shapes cannot be used together.
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(SCOPE, 8, 16, USE_B))) MatB8x16_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC16x16_t;

kernel void test_incompatible_muladd_shapes() {
    MatA_t       a;
    MatB8x16_t   b;
    MatC16x16_t  c;

    (void)coop_mat_mulAdd(a, b, c); // expected-error {{two cooperative matrices with incompatible shapes}}
}

// ---------------------------------------------------------------------------
// Cooperative matrix binary operators require supported operand types.
// ---------------------------------------------------------------------------
kernel void test_unsupported_coopmat_binary_operator() {
    MatA_t a;
    MatB_t b;

    (void)(a & b); // expected-error {{unsupported cooperative matrix binary operator}}
}

// ---------------------------------------------------------------------------
// Scalar cooperative-matrix operators are restricted to the supported
// operator/type combinations.
// ---------------------------------------------------------------------------
kernel void test_unsupported_coopmat_scalar_operator() {
    MatA_t a;

    (void)(a / 2.0f); // expected-error {{unsupported cooperative matrix scalar operator}}
}

// ---------------------------------------------------------------------------
// Cooperative matrix assignment requires compatible cooperative matrix types.
// ---------------------------------------------------------------------------
kernel void test_incompatible_coopmat_assignment() {
    MatA_t a;
    MatB_t b;

    a = b; // expected-error {{incompatible cooperative matrix types}}
}
