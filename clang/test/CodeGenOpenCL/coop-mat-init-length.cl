// Tests for coop_mat_init and coop_mat_length — CodeGen (IR shape).
//
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_cooperative_matrix \
// RUN:   -finclude-default-header -fdeclare-opencl-builtins -emit-llvm -O0 \
// RUN:   -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_cooperative_matrix : enable

// Type definitions.
typedef float __attribute__((coop_mat(memory_scope_sub_group,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_A)))           MatA_t;
typedef float __attribute__((coop_mat(memory_scope_sub_group,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_B)))           MatB_t;
typedef float __attribute__((coop_mat(memory_scope_sub_group,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_ACCUMULATOR))) MatC_t;
typedef int   __attribute__((coop_mat(memory_scope_sub_group,
                                      16, 16,
                                      CLK_COOPERATIVE_MATRIX_A)))           MatA_int_t;

// coop_mat_init with float scalar input.
kernel void test_init_float(global float *out) {
    MatA_t a;
    a = coop_mat_init(1.0f);
    (void)a;
}
// CHECK-LABEL: @{{.*}}test_init_float
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct{{[^(]*}}(float {{.*}}1.000000e+00)

// coop_mat_init with integer scalar input.
kernel void test_init_int(void) {
    MatA_int_t m;
    m = coop_mat_init(42);
    (void)m;
}
// CHECK-LABEL: @{{.*}}test_init_int
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct{{[^(]*}}(i32 {{.*}}42)

// coop_mat_init with zero initialisation.
kernel void test_init_zero(void) {
    MatC_t acc;
    acc = coop_mat_init(0.0f);
    (void)acc;
}
// CHECK-LABEL: @{{.*}}test_init_zero
// CHECK:         call spir_func target("spirv.CooperativeMatrixKHR"
// CHECK-SAME:    @__spirv_CompositeConstruct{{[^(]*}}(float {{.*}}0.000000e+00)

// coop_mat_length test.
kernel void test_length(global unsigned int *out) {
    MatA_t a;
    a = coop_mat_init(0.0f);
    unsigned int len = coop_mat_length(a);
    *out = len;
}
// CHECK-LABEL: @{{.*}}test_length
// CHECK:         call spir_func i32 @__spirv_CooperativeMatrixLengthKHR
// CHECK-SAME:    target("spirv.CooperativeMatrixKHR"
