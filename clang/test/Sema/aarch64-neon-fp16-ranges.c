// RUN: %clang_cc1 -triple arm64-linux-gnu -fallow-half-arguments-and-returns -target-feature +neon -target-feature +fullfp16 -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fallow-half-arguments-and-returns -target-feature +fullfp16 -target-feature +neon -ffreestanding -fsyntax-only -verify %s    

#include <arm_neon.h>
#include <arm_fp16.h>

void test_vcvt_f16_16(int16_t a){
  vcvth_n_f16_s16(a, 1);
  vcvth_n_f16_s16(a, 16);
  vcvth_n_f16_s16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_f16_s16(a, 17); // expected-error {{argument should be a value from 1 to 16}}

  vcvth_n_f16_u16(a, 1);
  vcvth_n_f16_u16(a, 16);
  vcvth_n_f16_u16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_f16_u16(a, 17); // expected-error {{argument should be a value from 1 to 16}}
}

void test_vcvt_f16_32(int32_t a){
  vcvth_n_f16_u32(a, 1);
  vcvth_n_f16_u32(a, 16);
  vcvth_n_f16_u32(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_f16_u32(a, 17); // expected-error {{argument should be a value from 1 to 16}}

  vcvth_n_f16_s32(a, 1);
  vcvth_n_f16_s32(a, 16);
  vcvth_n_f16_s32(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_f16_s32(a, 17); // expected-error {{argument should be a value from 1 to 16}}
}

void test_vcvt_f16_64(int64_t a){
  vcvth_n_f16_s64(a, 1);
  vcvth_n_f16_s64(a, 16);
  vcvth_n_f16_s64(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_f16_s64(a, 17); // expected-error {{argument should be a value from 1 to 16}}
}


void test_vcvt_su_f(float16_t a){
  vcvth_n_s16_f16(a, 1);
  vcvth_n_s16_f16(a, 16);
  vcvth_n_s16_f16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_s16_f16(a, 17); // expected-error {{argument should be a value from 1 to 16}}

  vcvth_n_s32_f16(a, 1);
  vcvth_n_s32_f16(a, 16);
  vcvth_n_s32_f16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_s32_f16(a, 17); // expected-error {{argument should be a value from 1 to 16}}

  vcvth_n_s64_f16(a, 1);
  vcvth_n_s64_f16(a, 16);
  vcvth_n_s64_f16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_s64_f16(a, 17); // expected-error {{argument should be a value from 1 to 16}}

  vcvth_n_u16_f16(a, 1);
  vcvth_n_u16_f16(a, 16);
  vcvth_n_u16_f16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_u16_f16(a, 17); // expected-error {{argument should be a value from 1 to 16}}

  vcvth_n_u32_f16(a, 1);
  vcvth_n_u32_f16(a, 16);
  vcvth_n_u32_f16(a, 0);  // expected-error {{argument should be a value from 1 to 16}}
  vcvth_n_u32_f16(a, 17); // expected-error {{argument should be a value from 1 to 16}}
}
