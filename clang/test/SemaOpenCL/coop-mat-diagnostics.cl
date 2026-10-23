// Negative tests for diagnostic paths.
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fdeclare-opencl-builtins -fsyntax-only \
// RUN:   -verify %s

#define SCOPE     memory_scope_sub_group
#define USE_A     CLK_COOPERATIVE_MATRIX_A
#define USE_B     CLK_COOPERATIVE_MATRIX_B
#define USE_C     CLK_COOPERATIVE_MATRIX_ACCUMULATOR
#define ROW_MAJOR CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR

// Use of cooperative matrix type without specifying pragma
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatX_t; // expected-error {{cooperative matrix types require OpenCL extension 'cl_khr_cooperative_matrix' to be enabled via pragma}}

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatA_t; // expected-note {{declared here}}
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_B))) MatB_t;
typedef int   __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_int_t;
typedef float   __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC_t;

// Invalid 'scope'
typedef float __attribute__((coop_mat(0, 16, 16, USE_A))) MatBadScope; // expected-error {{invalid 'scope' argument of cooperative matrix attribute}}

// Invalid 'use'
typedef float __attribute__((coop_mat(SCOPE, 16, 16, 99))) MatBadUse;  // expected-error {{invalid 'use' argument of cooperative matrix attribute}}

// Invalid element types.
typedef bool __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatBadElement_t; // expected-error {{'bool' is invalid cooperative matrix element type}}
typedef float * __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatPtrElem_t; // expected-error-re {{'{{.*}}float *' is invalid cooperative matrix element type}}
typedef float Arr4_t[4];
typedef Arr4_t __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatArrElem_t; // expected-error-re {{'Arr4_t'{{.*}} is invalid cooperative matrix element type}}
struct ElemS { float x; };
typedef struct ElemS __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatStructElem_t; // expected-error {{'struct ElemS' is invalid cooperative matrix element type}}
typedef float4 __attribute__((coop_mat(SCOPE, 16, 16, USE_A))) MatVecElem_t; // expected-error-re {{'float4'{{.*}} is invalid cooperative matrix element type}}

// Assignment of coop_mat_load result to a plain scalar
kernel void test_bad_assignment(__global float *ptr) {
    float bad;
	bad = coop_mat_load(ptr, ROW_MAJOR, 16); // expected-error {{builtin return value should be assigned to cooperative matrix type variable}}
    (void)bad;
}

// Kernel parameter cannot be of cooperative matrix type.
kernel void test_bad_kernel_param(MatA_t a) {} // expected-error {{cannot be used as the type of a kernel parameter}}

// Invalid memory layout for cooperative matrix load.
kernel void test_invalid_load_layout(__global float *ptr) {
    MatA_t a;
    a = coop_mat_load(ptr, 99, 16); // expected-error {{memory layout for cooperative matrix load/store should be CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR or CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR}}
}

// Invalid memory layout for cooperative matrix store.
kernel void test_invalid_store_layout(__global float *ptr) {
    MatA_t a;
    coop_mat_store(ptr, a, 99, 16); // expected-error {{memory layout for cooperative matrix load/store should be CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR or CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR}}
}

// Invalid stride type for cooperative matrix load.
kernel void test_invalid_load_stride(__global float *ptr) {
    MatA_t a;
    int stride = 16;
    a = coop_mat_load(ptr, ROW_MAJOR, stride); // expected-error {{stride argument for load/store of cooperative matrix must have type 'size_t'}}
}

// Invalid stride type for cooperative matrix store.
kernel void test_invalid_store_stride(__global float *ptr) {
    MatA_t a;
    int stride = 16;
    coop_mat_store(ptr, a, ROW_MAJOR, stride); // expected-error {{stride argument for load/store of cooperative matrix must have type 'size_t'}}
}

// Invalid value for fourth coop_mat_mulAdd argument.
kernel void test_invalid_muladd_matrix_operand_type() {
    MatA_t a;
    MatB_t b;
    MatC_t c;
    (void)coop_mat_mulAdd(a, b, c, 16.2f); // expected-error {{fourth argument of 'coop_mat_mulAdd' must have enum or integer type}}
}

// Tests for first three arguments of coop_mat_mulAdd.
kernel void test_invalid_muladd_argument(__global float *ptr) {
    MatA_t a;
    MatB_t b;
    (void)coop_mat_mulAdd(a, b, 1.0f); // expected-error {{argument must be a valid cooperative matrix}}
}

// Tests for first argument of coop_mat_mulAdd.
kernel void test_invalid_muladd_a_use(__global float *ptr) {
    MatB_t b;
    MatB_t b2;
    MatC_t c;
    (void)coop_mat_mulAdd(b, b2, c); // expected-error {{argument of cooperative matrix must have 'CLK_COOPERATIVE_MATRIX_A' use}}
}

// Tests for second argument of coop_mat_mulAdd.
kernel void test_invalid_muladd_b_use(__global float *ptr) {
    MatA_t a;
    MatA_t a2;
    MatC_t c;
    (void)coop_mat_mulAdd(a, a2, c); // expected-error {{argument of cooperative matrix must have 'CLK_COOPERATIVE_MATRIX_B' use}}
}

// Tests for third argument of coop_mat_mulAdd.
kernel void test_invalid_muladd_accumulator_use(__global float *ptr) {
    MatA_t a;
    MatB_t b;
    MatA_t c;
    (void)coop_mat_mulAdd(a, b, c); // expected-error {{argument of cooperative matrix must have 'CLK_COOPERATIVE_MATRIX_ACCUMULATOR' use}}
}

// Cooperative matrices with incompatible shapes cannot be used together.
typedef float __attribute__((coop_mat(SCOPE, 8, 16, USE_B))) MatB8x16_t;
typedef float __attribute__((coop_mat(SCOPE, 16, 16, USE_C))) MatC16x16_t;
kernel void test_incompatible_muladd_shapes() {
    MatA_t       a;
    MatB8x16_t   b;
    MatC16x16_t  c;
    (void)coop_mat_mulAdd(a, b, c); // expected-error {{two cooperative matrices with incompatible shapes}}
}

// Test of cooperative matrix binary operators.
kernel void test_unsupported_coopmat_binary_operator() {
    MatA_t a;
    MatB_t b;
    (void)(a & b); // expected-error {{unsupported cooperative matrix binary operator}}
}

// Test of scalar cooperative-matrix operators.
kernel void test_unsupported_coopmat_scalar_operator() {
    MatA_t a;
    (void)(a / 2.0f); // expected-error {{unsupported cooperative matrix scalar operator}}
}

// Test of cooperative matrix assignment.
kernel void test_incompatible_coopmat_assignment() {
    MatA_t a;
    MatB_t b;
    a = b; // expected-error {{incompatible cooperative matrix types}}
}

// Test of cooperative matrix cast assignment.
kernel void test_incompatible_coopmat_cast() {
    MatA_t a;
    MatB_t b;
    a = (MatA_t)b; // expected-error {{used type 'MatA_t' (aka 'float __attribute__((coop_mat(4, 16, 16, 0)))') where arithmetic or pointer type is required}}
}

// Test of coop_mat_init.
void test_coop_mat_init_invalid(void) {
  MatA_t a;
  struct S {
    int X;
  };
  struct S S;
  a = coop_mat_init(S); // expected-error {{invalid argument of cooperative matrix builtin}}
  float *ptr;
  a = coop_mat_init(ptr); // expected-error {{invalid argument of cooperative matrix builtin}}
  float arr[4];
  a = coop_mat_init(arr); // expected-error {{invalid argument of cooperative matrix builtin}}
  float4 arr4;
  a = coop_mat_init(arr4); // expected-error {{invalid argument of cooperative matrix builtin}}
  a = coop_mat_init((float *)(0x10000000)); // expected-error {{invalid argument of cooperative matrix builtin}}
}

// Test of coop_mat_length.
void test_coop_mat_length_invalid(void) {
  int Value = 0;
  uint len = coop_mat_length(Value); // expected-error {{argument must be a valid cooperative matrix}}
}
