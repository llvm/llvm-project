// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -verify -emit-llvm -o - %s

#include <arm_sve.h>

void test_features(svmfloat8_t zn, svmfloat8_t zm, mfloat8_t x, fpm_t fpm) {
  svcvt1_bf16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvt1_bf16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvt2_bf16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvt2_bf16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtlt1_bf16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvtlt1_bf16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtlt2_bf16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvtlt2_bf16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvt1_f16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvt1_f16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvt2_f16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvt2_f16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtlt1_f16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvtlt1_f16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtlt2_f16_mf8_fpm(zn, fpm);
  // expected-error@-1 {{'svcvtlt2_f16_mf8_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}

  svcvtn_mf8_bf16_x2_fpm(svcreate2(svundef_bf16(), svundef_bf16()), fpm);
  // expected-error@-1 {{'svcvtn_mf8_bf16_x2_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtn_mf8_f16_x2_fpm(svcreate2(svundef_f16(), svundef_f16()), fpm);
  // expected-error@-1 {{'svcvtn_mf8_f16_x2_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtnb_mf8_f32_x2_fpm(svcreate2(svundef_f32(), svundef_f32()), fpm);
  // expected-error@-1 {{'svcvtnb_mf8_f32_x2_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}
  svcvtnt_mf8_f32_x2_fpm(zn, svcreate2(svundef_f32(), svundef_f32()), fpm);
  // expected-error@-1 {{'svcvtnt_mf8_f32_x2_fpm' needs target feature (sve,sve2,fp8)|(sme,sme2,fp8)}}

  svdot_f32_mf8_fpm(svundef_f32(), zn, zm, fpm);
// expected-error@-1 {{'svdot_f32_mf8_fpm' needs target feature (sve,sve2,fp8dot4)|(sme,ssve-fp8dot4)}}
  svdot_n_f32_mf8_fpm(svundef_f32(), zn, x, fpm);
// expected-error@-1 {{'svdot_n_f32_mf8_fpm' needs target feature (sve,sve2,fp8dot4)|(sme,ssve-fp8dot4)}}
  svdot_f16_mf8_fpm(svundef_f16(), zn, zm, fpm);
// expected-error@-1 {{'svdot_f16_mf8_fpm' needs target feature (sve,sve2,fp8dot2)|(sme,ssve-fp8dot2)}}
  svdot_n_f16_mf8_fpm(svundef_f16(), zn, x, fpm);
// expected-error@-1 {{'svdot_n_f16_mf8_fpm' needs target feature (sve,sve2,fp8dot2)|(sme,ssve-fp8dot2)}}
  svdot_lane_f32_mf8_fpm(svundef_f32(), zn, zm, 3, fpm);
// expected-error@-1 {{'svdot_lane_f32_mf8_fpm' needs target feature (sve,sve2,fp8dot4)|(sme,ssve-fp8dot4)}}
  svdot_lane_f16_mf8_fpm(svundef_f16(), zn, zm, 7, fpm);
// expected-error@-1 {{'svdot_lane_f16_mf8_fpm' needs target feature (sve,sve2,fp8dot2)|(sme,ssve-fp8dot2)}}
}


void test_imm_range(svmfloat8_t zn, svmfloat8_t zm, fpm_t fpm) {
  svdot_lane_f32_mf8_fpm(svundef_f32(), zn, zm, -1, fpm);
// expected-error@-1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svdot_lane_f16_mf8_fpm(svundef_f16(), zn, zm, -1, fpm);
// expected-error@-1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
}