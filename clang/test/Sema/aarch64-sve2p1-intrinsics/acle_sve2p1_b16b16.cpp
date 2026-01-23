// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

#if defined __ARM_FEATURE_SME
#define MODE_ATTR __arm_streaming
#else
#define MODE_ATTR
#endif

__attribute__((target("sve-b16b16")))
void test_with_sve_b16b16(svbool_t pg, svbfloat16_t op1, svbfloat16_t op2, svbfloat16_t op3) MODE_ATTR
{
  svclamp_bf16(op1, op2, op3);
  svadd_bf16_m(pg, op1, op2);
  svmax_bf16_m(pg, op1, op2);
  svmaxnm_bf16_m(pg, op1, op2);
  svmin_bf16_m(pg, op1, op2);
  svminnm_bf16_m(pg, op1, op2);
  svmla_lane_bf16(op1, op2, op3, 1);
  svmla_bf16_m(pg, op1, op2, op3);
  svmls_bf16_m(pg, op1, op2, op3);
  svmul_lane_bf16(op1, op2, 1);
  svmul_bf16_m(pg, op1, op2);
  svsub_bf16_m(pg, op1, op2);
}

void test_no_sve_b16b16(svbool_t pg, svbfloat16_t op1, svbfloat16_t op2, svbfloat16_t op3) MODE_ATTR
{
  // expected-error@+1 {{'svclamp_bf16' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svclamp_bf16(op1, op2, op3);
  // expected-error@+1 {{'svadd_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svadd_bf16_m(pg, op1, op2);
  // expected-error@+1 {{'svmax_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmax_bf16_m(pg, op1, op2);
  // expected-error@+1 {{'svmaxnm_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmaxnm_bf16_m(pg, op1, op2);
  // expected-error@+1 {{'svmin_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmin_bf16_m(pg, op1, op2);
  // expected-error@+1 {{'svminnm_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svminnm_bf16_m(pg, op1, op2);
  // expected-error@+1 {{'svmla_lane_bf16' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmla_lane_bf16(op1, op2, op3, 1);
  // expected-error@+1 {{'svmla_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmla_bf16_m(pg, op1, op2, op3);
  // expected-error@+1 {{'svmls_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmls_bf16_m(pg, op1, op2, op3);
  // expected-error@+1 {{'svmul_lane_bf16' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmul_lane_bf16(op1, op2, 1);
  // expected-error@+1 {{'svmul_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svmul_bf16_m(pg, op1, op2);
  // expected-error@+1 {{'svsub_bf16_m' needs target feature (sve,sve-b16b16)|(sme,sme2,sve-b16b16)}}
  svsub_bf16_m(pg, op1, op2);
}
