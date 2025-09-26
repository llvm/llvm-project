// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

#if defined __ARM_FEATURE_SME
#define MODE_ATTR __arm_streaming
#else
#define MODE_ATTR
#endif

__attribute__((target("bf16")))
void test_bf16(svbool_t pg, svfloat32_t svf32, svbfloat16_t svbf16, bfloat16_t bf16) MODE_ATTR
{
  svbfdot_f32(svf32, svbf16, svbf16);
  svbfdot_n_f32(svf32, svbf16, bf16);
  svbfdot_lane_f32(svf32, svbf16, svbf16, 0);

  svbfmlalb_f32(svf32, svbf16, svbf16);
  svbfmlalb_n_f32(svf32, svbf16, bf16);
  svbfmlalb_lane_f32(svf32, svbf16, svbf16, 0);

  svbfmlalt_f32(svf32, svbf16, svbf16);
  svbfmlalt_n_f32(svf32, svbf16, bf16);
  svbfmlalt_lane_f32(svf32, svbf16, svbf16, 0);

  svcvt_bf16_f32_m(svbf16, pg, svf32);
  svcvt_bf16_f32_x(pg, svf32);
  svcvt_bf16_f32_z(pg, svf32);

  svcvtnt_bf16_f32_m(svbf16, pg, svf32);
  svcvtnt_bf16_f32_x(svbf16, pg, svf32);
}

void test_no_bf16(svbool_t pg, svfloat32_t svf32, svbfloat16_t svbf16, bfloat16_t bf16) MODE_ATTR
{
  // expected-error@+1 {{'svbfdot_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfdot_f32(svf32, svbf16, svbf16);
  // expected-error@+1 {{'svbfdot_n_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfdot_n_f32(svf32, svbf16, bf16);
  // expected-error@+1 {{'svbfdot_lane_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfdot_lane_f32(svf32, svbf16, svbf16, 0);

  // expected-error@+1 {{'svbfmlalb_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfmlalb_f32(svf32, svbf16, svbf16);
  // expected-error@+1 {{'svbfmlalb_n_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfmlalb_n_f32(svf32, svbf16, bf16);
  // expected-error@+1 {{'svbfmlalb_lane_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfmlalb_lane_f32(svf32, svbf16, svbf16, 0);

  // expected-error@+1 {{'svbfmlalt_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfmlalt_f32(svf32, svbf16, svbf16);
  // expected-error@+1 {{'svbfmlalt_n_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfmlalt_n_f32(svf32, svbf16, bf16);
  // expected-error@+1 {{'svbfmlalt_lane_f32' needs target feature (sve,bf16)|(sme,bf16)}}
  svbfmlalt_lane_f32(svf32, svbf16, svbf16, 0);

  // expected-error@+1 {{'svcvt_bf16_f32_m' needs target feature (sve,bf16)|(sme,bf16)}}
  svcvt_bf16_f32_m(svbf16, pg, svf32);
  // expected-error@+1 {{'svcvt_bf16_f32_x' needs target feature (sve,bf16)|(sme,bf16)}}
  svcvt_bf16_f32_x(pg, svf32);
  // expected-error@+1 {{'svcvt_bf16_f32_z' needs target feature (sve,bf16)|(sme,bf16)}}
  svcvt_bf16_f32_z(pg, svf32);

  // expected-error@+1 {{'svcvtnt_bf16_f32_m' needs target feature (sve,bf16)|(sme,bf16)}}
  svcvtnt_bf16_f32_m(svbf16, pg, svf32);
  // NOTE: svcvtnt_bf16_f32_x is a macro that expands to svcvtnt_bf16_f32_m.
  // expected-error@+1 {{'svcvtnt_bf16_f32_m' needs target feature (sve,bf16)|(sme,bf16)}}
  svcvtnt_bf16_f32_x(svbf16, pg, svf32);
}
