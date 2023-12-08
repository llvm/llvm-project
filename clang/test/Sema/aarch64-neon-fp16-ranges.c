// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -target-feature +fullfp16 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +fullfp16 -target-feature +neon -ffreestanding -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>
#include <arm_fp16.h>

void test_vcvt_f16_16(int16_t a){
  vcvth_n_f16_s16(a, 1);
  vcvth_n_f16_s16(a, 16);
  vcvth_n_f16_s16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_f16_s16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  vcvth_n_f16_u16(a, 1);
  vcvth_n_f16_u16(a, 16);
  vcvth_n_f16_u16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_f16_u16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcvt_f16_32(int32_t a){
  vcvth_n_f16_u32(a, 1);
  vcvth_n_f16_u32(a, 16);
  vcvth_n_f16_u32(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_f16_u32(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  vcvth_n_f16_s32(a, 1);
  vcvth_n_f16_s32(a, 16);
  vcvth_n_f16_s32(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_f16_s32(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcvt_f16_64(int64_t a){
  vcvth_n_f16_s64(a, 1);
  vcvth_n_f16_s64(a, 16);
  vcvth_n_f16_s64(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_f16_s64(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}


void test_vcvt_su_f(float16_t a){
  vcvth_n_s16_f16(a, 1);
  vcvth_n_s16_f16(a, 16);
  vcvth_n_s16_f16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_s16_f16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  vcvth_n_s32_f16(a, 1);
  vcvth_n_s32_f16(a, 16);
  vcvth_n_s32_f16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_s32_f16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  vcvth_n_s64_f16(a, 1);
  vcvth_n_s64_f16(a, 16);
  vcvth_n_s64_f16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_s64_f16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  vcvth_n_u16_f16(a, 1);
  vcvth_n_u16_f16(a, 16);
  vcvth_n_u16_f16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_u16_f16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  vcvth_n_u32_f16(a, 1);
  vcvth_n_u32_f16(a, 16);
  vcvth_n_u32_f16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcvth_n_u32_f16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}
