// RUN: %clang_cc1 %s -triple spir-unknown-unknown -finclude-default-header -cl-std=CL1.2 -emit-llvm -o - -O0 | FileCheck %s

#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
void test_int8(uchar4 ua, uchar4 ub, char4 sa, char4 sb) {
    uint ur = arm_dot(ua, ub);
    // CHECK: call spir_func i32 @_Z7arm_dotDv4_hS_
    int sr = arm_dot(sa, sb);
    // CHECK: call spir_func i32 @_Z7arm_dotDv4_cS_
}
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : disable

#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
void test_accumulate_int8(uchar4 ua, uchar4 ub, uint uc, char4 sa, char4 sb, int c) {
    uint ur = arm_dot_acc(ua, ub, uc);
    // CHECK: call spir_func i32 @_Z11arm_dot_accDv4_hS_j
    int sr = arm_dot_acc(sa, sb, c);
    // CHECK: call spir_func i32 @_Z11arm_dot_accDv4_cS_i
}
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : disable

#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int16 : enable
void test_accumulate_int16(ushort2 ua, ushort2 ub, uint uc, short2 sa, short2 sb, int c) {
    uint ur = arm_dot_acc(ua, ub, uc);
    // CHECK: call spir_func i32 @_Z11arm_dot_accDv2_tS_j
    int sr = arm_dot_acc(sa, sb, c);
    // CHECK: call spir_func i32 @_Z11arm_dot_accDv2_sS_i
}
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int16 : disable

#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_saturate_int8 : enable
void test_accumulate_saturate_int8(uchar4 ua, uchar4 ub, uint uc, char4 sa, char4 sb, int c) {
    uint ur = arm_dot_acc_sat(ua, ub, uc);
    // CHECK: call spir_func i32 @_Z15arm_dot_acc_satDv4_hS_j
    int sr = arm_dot_acc_sat(sa, sb, c);
    // CHECK: call spir_func i32 @_Z15arm_dot_acc_satDv4_cS_i
}
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_saturate_int8 : disable

