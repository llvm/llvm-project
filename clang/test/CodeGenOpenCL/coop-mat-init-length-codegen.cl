// Tests for coop_mat_init and coop_mat_length — CodeGen (IR shape).
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_ext_kernel_cooperative_matrix \
// RUN:   -finclude-default-header -emit-llvm -O0 -o - %s \
// RUN:   | FileCheck %s

// ---------------------------------------------------------------------------
// Type definitions.
// ---------------------------------------------------------------------------
typedef float __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_A)))           MatA_t;
typedef float __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_ACCUMULATOR))) MatC_t;
typedef int   __attribute__((coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_A)))           MatA_int_t;

// ---------------------------------------------------------------------------
// 1. coop_mat_init — float scalar.
//    Expected IR: call __spirv_CompositeConstruct with the scalar value,
//    returning a spirv.CooperativeMatrixKHR TargetExtType.
// ---------------------------------------------------------------------------

// CHECK-LABEL: @{{.*}}test_init_float
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct

kernel void test_init_float(global float *out) {
    MatA_t a;
    a = coop_mat_init(1.0f);
    (void)a;
}

// ---------------------------------------------------------------------------
// 2. coop_mat_init — integer scalar.
//    Expected IR: call __spirv_CompositeConstruct with an i32 value.
// ---------------------------------------------------------------------------

// CHECK-LABEL: @{{.*}}test_init_int
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct

kernel void test_init_int(void) {
    MatA_int_t m;
    m = coop_mat_init(42);
    (void)m;
}

// ---------------------------------------------------------------------------
// 3. coop_mat_init — zero initialisation.
//    Expected IR: call __spirv_CompositeConstruct with 0.0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: @{{.*}}test_init_zero
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct_{{.*}}(float {{.*}}0

kernel void test_init_zero(void) {
    MatC_t acc;
    acc = coop_mat_init(0.0f);
    (void)acc;
}

// ---------------------------------------------------------------------------
// 4. coop_mat_length — returns unsigned int.
//    Expected IR: call __spirv_CooperativeMatrixLengthKHR, result stored into
//    an i32 alloca.
// ---------------------------------------------------------------------------

// CHECK-LABEL: @{{.*}}test_length
// CHECK:         call spir_func i32 @__spirv_CooperativeMatrixLengthKHR
// CHECK-SAME:    target("spirv.CooperativeMatrixKHR"

kernel void test_length(global unsigned int *out) {
    MatA_t a;
    a = coop_mat_init(0.0f);
    unsigned int len = coop_mat_length(a);
    *out = len;
}

// ---------------------------------------------------------------------------
// 5. coop_mat_init then coop_mat_length — combined flow.
//    Verifies the TargetExtType produced by init flows into length unchanged.
// ---------------------------------------------------------------------------

// CHECK-LABEL: @{{.*}}test_init_then_length
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct
// CHECK:         call spir_func i32 @__spirv_CooperativeMatrixLengthKHR

kernel void test_init_then_length(global unsigned int *out) {
    MatA_t a;
    a = coop_mat_init(2.0f);
    *out = coop_mat_length(a);
}

// ---------------------------------------------------------------------------
// 6. coop_mat_length — calling convention is SPIR_FUNC.
//    The call must NOT be a plain 'call' — it must be 'call spir_func'.
// ---------------------------------------------------------------------------

// CHECK-LABEL: @{{.*}}test_length_cc
// CHECK:        call spir_func i32 @__spirv_CooperativeMatrixLengthKHR

kernel void test_length_cc(global unsigned int *out) {
    MatC_t acc;
    acc = coop_mat_init(0.0f);
    *out = coop_mat_length(acc);
}
