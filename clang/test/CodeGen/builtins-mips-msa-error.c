// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -fsyntax-only %s \
// RUN:            -target-feature +msa -target-feature +fp64 \
// RUN:            -verify -mfloat-abi hard -o - 2>&1

#include <msa.h>

void test(void) {
  v16i8 v16i8_a = (v16i8) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  v16i8 v16i8_b = (v16i8) {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  v16i8 v16i8_r;
  v8i16 v8i16_a = (v8i16) {0, 1, 2, 3, 4, 5, 6, 7};
  v8i16 v8i16_b = (v8i16) {8, 9, 10, 11, 12, 13, 14, 15};
  v8i16 v8i16_r;
  v4i32 v4i32_a = (v4i32) {0, 1, 2, 3};
  v4i32 v4i32_b = (v4i32) {4, 5, 6, 7};
  v4i32 v4i32_r;
  v2i64 v2i64_a = (v2i64) {0, 1};
  v2i64 v2i64_b = (v2i64) {3, 4};
  v2i64 v2i64_r;

  v16u8 v16u8_a = (v16u8) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  v16u8 v16u8_r;
  v8u16 v8u16_a = (v8u16) {0, 1, 2, 3, 4, 5, 6, 7};
  v8u16 v8u16_r;
  v4u32 v4u32_a = (v4u32) {0, 1, 2, 3};
  v4u32 v4u32_r;
  v2u64 v2u64_a = (v2u64) {0, 1};
  v2u64 v2u64_r;

  int int_r;
  long long ll_r;

  v16u8_r = __msa_addvi_b(v16u8_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v8u16_r = __msa_addvi_h(v8u16_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v4u32_r = __msa_addvi_w(v4u32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2u64_r = __msa_addvi_d(v2u64_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  v16i8_r = __msa_andi_b(v16i8_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v8i16_r = __msa_andi_b(v8i16_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v4i32_r = __msa_andi_b(v4i32_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v2i64_r = __msa_andi_b(v2i64_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bclri_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_bclri_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_bclri_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_bclri_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_binsli_b(v16i8_r, v16i8_a, 8);     // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_binsli_h(v8i16_r, v8i16_a, 16);    // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_binsli_w(v4i32_r, v4i32_a, 32);    // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_binsli_d(v2i64_r, v2i64_a, 64);    // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_binsri_b(v16i8_r, v16i8_a, 8);     // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_binsri_h(v8i16_r, v8i16_a, 16);    // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_binsri_w(v4i32_r, v4i32_a, 32);    // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_binsri_d(v2i64_r, v2i64_a, 64);    // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_bmnzi_b(v16i8_r, v16i8_a, 256);    // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bmzi_b(v16i8_r, v16i8_a, 256);     // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bnegi_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_bnegi_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_bnegi_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_bnegi_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_bseli_b(v16i8_r, v16i8_a, 256);    // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bseti_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_bseti_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_bseti_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_bseti_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_ceqi_b(v16i8_a, 16);               // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_ceqi_h(v8i16_a, 16);               // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_ceqi_w(v4i32_a, 16);               // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_ceqi_d(v2i64_a, 16);               // expected-error {{argument value 16 is outside the valid range [-16, 15]}}

  int_r = __msa_cfcmsa(32);                          // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  v16i8_r = __msa_clei_s_b(v16i8_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_clei_s_h(v8i16_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_clei_s_w(v4i32_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_clei_s_d(v2i64_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_clei_u_b(v16u8_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v8u16_r = __msa_clei_u_h(v8u16_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v4u32_r = __msa_clei_u_w(v4u32_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2u64_r = __msa_clei_u_d(v2u64_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  v16i8_r = __msa_clti_s_b(v16i8_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_clti_s_h(v8i16_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_clti_s_w(v4i32_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_clti_s_d(v2i64_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_clti_u_b(v16u8_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v8u16_r = __msa_clti_u_h(v8u16_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v4u32_r = __msa_clti_u_w(v4u32_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2u64_r = __msa_clti_u_d(v2u64_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  int_r = __msa_copy_s_b(v16i8_a, 16);               // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  int_r = __msa_copy_s_h(v8i16_a, 8);                // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  int_r = __msa_copy_s_w(v4i32_a, 4);                // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  ll_r  = __msa_copy_s_d(v2i64_a, 2);                // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  int_r = __msa_copy_u_b(v16u8_a, 16);               // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  int_r = __msa_copy_u_h(v8u16_a, 8);                // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  int_r = __msa_copy_u_w(v4u32_a, 4);                // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  ll_r  = __msa_copy_u_d(v2i64_a, 2);                // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  __builtin_msa_ctcmsa(32, 777);                     // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  v16i8_r = __msa_insve_b(v16i8_r, 16, v16i8_a);     // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v8i16_r = __msa_insve_h(v8i16_r, 8, v8i16_a);      // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v4i32_r = __msa_insve_w(v4i32_r, 4, v4i32_a);      // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  v2i64_r = __msa_insve_d(v2i64_r, 2, v2i64_a);      // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  v8i16_r = __msa_ld_h(&v8i16_a, 77);                // expected-error {{argument should be a multiple of 2}}
  v4i32_r = __msa_ld_w(&v4i32_a, 14);                // expected-error {{argument should be a multiple of 4}}
  v2i64_r = __msa_ld_d(&v2i64_a, 23);                // expected-error {{argument should be a multiple of 8}}

  v16i8_r = __msa_ld_b(&v16i8_a, 512);               // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  v8i16_r = __msa_ld_h(&v8i16_a, 1024);              // expected-error {{argument value 1024 is outside the valid range [-1024, 1022]}}
  v4i32_r = __msa_ld_w(&v4i32_a, 2048);              // expected-error {{argument value 2048 is outside the valid range [-2048, 2044]}}
  v2i64_r = __msa_ld_d(&v2i64_a, 4096);              // expected-error {{argument value 4096 is outside the valid range [-4096, 4088]}}

  v16i8_r = __msa_ldi_b(256);                        // expected-error {{argument value 256 is outside the valid range [-128, 255]}}
  v8i16_r = __msa_ldi_h(512);                        // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  v4i32_r = __msa_ldi_w(512);                        // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  v2i64_r = __msa_ldi_d(512);                        // expected-error {{argument value 512 is outside the valid range [-512, 511]}}

  v16i8_r = __msa_maxi_s_b(v16i8_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_maxi_s_h(v8i16_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_maxi_s_w(v4i32_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_maxi_s_d(v2i64_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_maxi_u_b(v16u8_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v8u16_r = __msa_maxi_u_h(v8u16_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v4u32_r = __msa_maxi_u_w(v4u32_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2u64_r = __msa_maxi_u_d(v2u64_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  v16i8_r = __msa_mini_s_b(v16i8_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_mini_s_h(v8i16_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_mini_s_w(v4i32_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_mini_s_d(v2i64_a, 16);             // expected-error {{argument value 16 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_mini_u_b(v16u8_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v8u16_r = __msa_mini_u_h(v8u16_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v4u32_r = __msa_mini_u_w(v4u32_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2u64_r = __msa_mini_u_d(v2u64_a, 32);             // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  v16i8_r = __msa_nori_b(v16i8_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_ori_b(v16i8_a, 256);               // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_sat_s_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_sat_s_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_sat_s_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_sat_s_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_sat_u_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_sat_u_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_sat_u_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_sat_u_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_shf_b(v16i8_a, 256);               // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v8i16_r = __msa_shf_h(v8i16_a, 256);               // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v4i32_r = __msa_shf_w(v4i32_a, 256);               // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16i8_r = __msa_sldi_b(v16i8_r, v16i8_a, 16);      // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v8i16_r = __msa_sldi_h(v8i16_r, v8i16_a, 8);       // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v4i32_r = __msa_sldi_w(v4i32_r, v4i32_a, 4);       // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  v2i64_r = __msa_sldi_d(v2i64_r, v2i64_a, 2);       // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  v16i8_r = __msa_slli_b(v16i8_a, 8);                // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_slli_h(v8i16_a, 16);               // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_slli_w(v4i32_a, 32);               // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_slli_d(v2i64_a, 64);               // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_splati_b(v16i8_a, 16);             // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v8i16_r = __msa_splati_h(v8i16_a, 8);              // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v4i32_r = __msa_splati_w(v4i32_a, 4);              // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  v2i64_r = __msa_splati_d(v2i64_a, 2);              // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  v16i8_r = __msa_srai_b(v16i8_a, 8);                // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srai_h(v8i16_a, 16);               // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srai_w(v4i32_a, 32);               // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srai_d(v2i64_a, 64);               // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_srari_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srari_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srari_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srari_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_srli_b(v16i8_a, 8);                // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srli_h(v8i16_a, 16);               // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srli_w(v4i32_a, 32);               // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srli_d(v2i64_a, 64);               // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  v16i8_r = __msa_srlri_b(v16i8_a, 8);               // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srlri_h(v8i16_a, 16);              // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srlri_w(v4i32_a, 32);              // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srlri_d(v2i64_a, 64);              // expected-error {{argument value 64 is outside the valid range [0, 63]}}

  __msa_st_h(v8i16_b, &v8i16_a, 51);                // expected-error {{argument should be a multiple of 2}}
  __msa_st_w(v4i32_b, &v4i32_a, 51);                // expected-error {{argument should be a multiple of 4}}
  __msa_st_d(v2i64_b, &v2i64_a, 12);                // expected-error {{argument should be a multiple of 8}}

  __msa_st_b(v16i8_b, &v16i8_a, 512);                // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  __msa_st_h(v8i16_b, &v8i16_a, 1024);                // expected-error {{argument value 1024 is outside the valid range [-1024, 1022]}}
  __msa_st_w(v4i32_b, &v4i32_a, 2048);                // expected-error {{argument value 2048 is outside the valid range [-2048, 2044]}}
  __msa_st_d(v2i64_b, &v2i64_a, 4096);                // expected-error {{argument value 4096 is outside the valid range [-4096, 4088]}}

  v16i8_r = __msa_subvi_b(v16i8_a, 256);             // expected-error {{argument value 256 is outside the valid range [0, 31]}}
  v8i16_r = __msa_subvi_h(v8i16_a, 256);             // expected-error {{argument value 256 is outside the valid range [0, 31]}}
  v4i32_r = __msa_subvi_w(v4i32_a, 256);             // expected-error {{argument value 256 is outside the valid range [0, 31]}}
  v2i64_r = __msa_subvi_d(v2i64_a, 256);             // expected-error {{argument value 256 is outside the valid range [0, 31]}}

  v16i8_r = __msa_xori_b(v16i8_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v8i16_r = __msa_xori_b(v8i16_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v4i32_r = __msa_xori_b(v4i32_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v2i64_r = __msa_xori_b(v2i64_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  v16u8_r = __msa_xori_b(v16u8_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v8u16_r = __msa_xori_b(v8u16_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v4u32_r = __msa_xori_b(v4u32_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  v2u64_r = __msa_xori_b(v2u64_a, 256);              // expected-error {{argument value 256 is outside the valid range [0, 255]}}

  // Test the lower bounds

  v16u8_r = __msa_addvi_b(v16u8_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v8u16_r = __msa_addvi_h(v8u16_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v4u32_r = __msa_addvi_w(v4u32_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2u64_r = __msa_addvi_d(v2u64_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}

  v16i8_r = __msa_andi_b(v16i8_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v8i16_r = __msa_andi_b(v8i16_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v4i32_r = __msa_andi_b(v4i32_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v2i64_r = __msa_andi_b(v2i64_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bclri_b(v16i8_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  v8i16_r = __msa_bclri_h(v8i16_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  v4i32_r = __msa_bclri_w(v4i32_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_bclri_d(v2i64_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}

  v16i8_r = __msa_binsli_b(v16i8_r, v16i8_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  v8i16_r = __msa_binsli_h(v8i16_r, v8i16_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  v4i32_r = __msa_binsli_w(v4i32_r, v4i32_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_binsli_d(v2i64_r, v2i64_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}

  v16i8_r = __msa_binsri_b(v16i8_r, v16i8_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  v8i16_r = __msa_binsri_h(v8i16_r, v8i16_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  v4i32_r = __msa_binsri_w(v4i32_r, v4i32_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_binsri_d(v2i64_r, v2i64_a, -1);    // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}

  v16i8_r = __msa_bmnzi_b(v16i8_r, v16i8_a, -1);     // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bmzi_b(v16i8_r, v16i8_a, -1);      // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bnegi_b(v16i8_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  v8i16_r = __msa_bnegi_h(v8i16_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  v4i32_r = __msa_bnegi_w(v4i32_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_bnegi_d(v2i64_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}

  v16i8_r = __msa_bseli_b(v16i8_r, v16i8_a, -1);     // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_bseti_b(v16i8_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  v8i16_r = __msa_bseti_h(v8i16_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  v4i32_r = __msa_bseti_w(v4i32_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_bseti_d(v2i64_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}

  v16i8_r = __msa_ceqi_b(v16i8_a, -17);              // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_ceqi_h(v8i16_a, -17);              // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_ceqi_w(v4i32_a, -17);              // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_ceqi_d(v2i64_a, -17);              // expected-error {{argument value -17 is outside the valid range [-16, 15]}}

  v16i8_r = __msa_clei_s_b(v16i8_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_clei_s_h(v8i16_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_clei_s_w(v4i32_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_clei_s_d(v2i64_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_clei_u_b(v16u8_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v8u16_r = __msa_clei_u_h(v8u16_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v4u32_r = __msa_clei_u_w(v4u32_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2u64_r = __msa_clei_u_d(v2u64_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}

  v16i8_r = __msa_clti_s_b(v16i8_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_clti_s_h(v8i16_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_clti_s_w(v4i32_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_clti_s_d(v2i64_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_clti_u_b(v16u8_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v8u16_r = __msa_clti_u_h(v8u16_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v4u32_r = __msa_clti_u_w(v4u32_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2u64_r = __msa_clti_u_d(v2u64_a, -1);             // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}

  int_r = __msa_copy_s_b(v16i8_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  int_r = __msa_copy_s_h(v8i16_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  int_r = __msa_copy_s_w(v4i32_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  ll_r  = __msa_copy_s_d(v2i64_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}

  int_r = __msa_copy_u_b(v16u8_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  int_r = __msa_copy_u_h(v8u16_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  int_r = __msa_copy_u_w(v4u32_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  ll_r  = __msa_copy_u_d(v2i64_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}

  v16i8_r = __msa_insve_b(v16i8_r, 16, v16i8_a);     // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  v8i16_r = __msa_insve_h(v8i16_r, 8, v8i16_a);      // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  v4i32_r = __msa_insve_w(v4i32_r, 4, v4i32_a);      // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  v2i64_r = __msa_insve_d(v2i64_r, 2, v2i64_a);      // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  v16i8_r = __msa_ld_b(&v16i8_a, -513);              // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  v8i16_r = __msa_ld_h(&v8i16_a, -1028);              // expected-error {{argument value -1028 is outside the valid range [-1024, 1022]}}
  v4i32_r = __msa_ld_w(&v4i32_a, -2052);              // expected-error {{argument value -2052 is outside the valid range [-2048, 2044]}}
  v2i64_r = __msa_ld_d(&v2i64_a, -4104);              // expected-error {{argument value -4104 is outside the valid range [-4096, 4088]}}

  v16i8_r = __msa_ldi_b(-129);                       // expected-error {{argument value -129 is outside the valid range [-128, 255]}}
  v8i16_r = __msa_ldi_h(-513);                       // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  v4i32_r = __msa_ldi_w(-513);                       // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  v2i64_r = __msa_ldi_d(-513);                       // expected-error {{argument value -513 is outside the valid range [-512, 511]}}

  v16i8_r = __msa_maxi_s_b(v16i8_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_maxi_s_h(v8i16_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_maxi_s_w(v4i32_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_maxi_s_d(v2i64_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_maxi_u_b(v16u8_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  v8u16_r = __msa_maxi_u_h(v8u16_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  v4u32_r = __msa_maxi_u_w(v4u32_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  v2u64_r = __msa_maxi_u_d(v2u64_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}

  v16i8_r = __msa_mini_s_b(v16i8_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v8i16_r = __msa_mini_s_h(v8i16_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v4i32_r = __msa_mini_s_w(v4i32_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  v2i64_r = __msa_mini_s_d(v2i64_a, -17);            // expected-error {{argument value -17 is outside the valid range [-16, 15]}}

  v16u8_r = __msa_mini_u_b(v16u8_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  v8u16_r = __msa_mini_u_h(v8u16_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  v4u32_r = __msa_mini_u_w(v4u32_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  v2u64_r = __msa_mini_u_d(v2u64_a, -1);             // expected-error {{argument value -1 is outside the valid range [0, 31]}}

  v16i8_r = __msa_nori_b(v16i8_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_ori_b(v16i8_a, -1);                // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_sat_s_b(v16i8_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  v8i16_r = __msa_sat_s_h(v8i16_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  v4i32_r = __msa_sat_s_w(v4i32_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_sat_s_d(v2i64_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}

  v16i8_r = __msa_sat_u_b(v16i8_a, -8);              // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v8i16_r = __msa_sat_u_h(v8i16_a, -17);             // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v4i32_r = __msa_sat_u_w(v4i32_a, -32);             // expected-error {{argument value 4294967264 is outside the valid range [0, 31]}}
  v2i64_r = __msa_sat_u_d(v2i64_a, -64);             // expected-error {{argument value 4294967232 is outside the valid range [0, 63]}}

  v16i8_r = __msa_shf_b(v16i8_a, -1);                // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v8i16_r = __msa_shf_h(v8i16_a, -1);                // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v4i32_r = __msa_shf_w(v4i32_a, -1);                // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16i8_r = __msa_sldi_b(v16i8_r, v16i8_a, -17);     // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v8i16_r = __msa_sldi_h(v8i16_r, v8i16_a, -8);      // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v4i32_r = __msa_sldi_w(v4i32_r, v4i32_a, -4);      // expected-error {{argument value 4294967292 is outside the valid range [0, 3]}}
  v2i64_r = __msa_sldi_d(v2i64_r, v2i64_a, -2);      // expected-error {{argument value 4294967294 is outside the valid range [0, 1]}}

  v16i8_r = __msa_slli_b(v16i8_a, -8);               // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v8i16_r = __msa_slli_h(v8i16_a, -17);              // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v4i32_r = __msa_slli_w(v4i32_a, -32);              // expected-error {{argument value 4294967264 is outside the valid range [0, 31]}}
  v2i64_r = __msa_slli_d(v2i64_a, -64);              // expected-error {{argument value 4294967232 is outside the valid range [0, 63]}}

  v16i8_r = __msa_splati_b(v16i8_a, -17);            // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v8i16_r = __msa_splati_h(v8i16_a, -8);             // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v4i32_r = __msa_splati_w(v4i32_a, -4);             // expected-error {{argument value 4294967292 is outside the valid range [0, 3]}}
  v2i64_r = __msa_splati_d(v2i64_a, -2);             // expected-error {{argument value 4294967294 is outside the valid range [0, 1]}}

  v16i8_r = __msa_srai_b(v16i8_a, -8);               // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srai_h(v8i16_a, -17);              // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srai_w(v4i32_a, -32);              // expected-error {{argument value 4294967264 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srai_d(v2i64_a, -64);              // expected-error {{argument value 4294967232 is outside the valid range [0, 63]}}

  v16i8_r = __msa_srari_b(v16i8_a, -8);              // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srari_h(v8i16_a, -17);             // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srari_w(v4i32_a, -32);             // expected-error {{argument value 4294967264 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srari_d(v2i64_a, -64);             // expected-error {{argument value 4294967232 is outside the valid range [0, 63]}}

  v16i8_r = __msa_srli_b(v16i8_a, -8);               // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srli_h(v8i16_a, -17);              // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srli_w(v4i32_a, -32);              // expected-error {{argument value 4294967264 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srli_d(v2i64_a, -64);              // expected-error {{argument value 4294967232 is outside the valid range [0, 63]}}

  v16i8_r = __msa_srlri_b(v16i8_a, -8);              // expected-error {{argument value 4294967288 is outside the valid range [0, 7]}}
  v8i16_r = __msa_srlri_h(v8i16_a, -17);             // expected-error {{argument value 4294967279 is outside the valid range [0, 15]}}
  v4i32_r = __msa_srlri_w(v4i32_a, -32);             // expected-error {{argument value 4294967264 is outside the valid range [0, 31]}}
  v2i64_r = __msa_srlri_d(v2i64_a, -64);             // expected-error {{argument value 4294967232 is outside the valid range [0, 63]}}

  __msa_st_b(v16i8_b, &v16i8_a, -513);               // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  __msa_st_h(v8i16_b, &v8i16_a, -1025);               // expected-error {{argument value -1025 is outside the valid range [-1024, 1022]}}
  __msa_st_w(v4i32_b, &v4i32_a, -2049);               // expected-error {{argument value -2049 is outside the valid range [-2048, 2044]}}
  __msa_st_d(v2i64_b, &v2i64_a, -4097);               // expected-error {{argument value -4097 is outside the valid range [-4096, 4088]}}

  v16i8_r = __msa_subvi_b(v16i8_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v8i16_r = __msa_subvi_h(v8i16_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v4i32_r = __msa_subvi_w(v4i32_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  v2i64_r = __msa_subvi_d(v2i64_a, -1);              // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}

  v16i8_r = __msa_xori_b(v16i8_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v8i16_r = __msa_xori_b(v8i16_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v4i32_r = __msa_xori_b(v4i32_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v2i64_r = __msa_xori_b(v2i64_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

  v16u8_r = __msa_xori_b(v16u8_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v8u16_r = __msa_xori_b(v8u16_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v4u32_r = __msa_xori_b(v4u32_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  v2u64_r = __msa_xori_b(v2u64_a, -1);               // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}

}
