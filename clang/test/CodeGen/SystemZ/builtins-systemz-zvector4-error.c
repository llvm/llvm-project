// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch14 -triple s390x-linux-gnu \
// RUN: -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -fsyntax-only -verify %s

#include <vecintrin.h>

volatile vector signed char vsc;
volatile vector signed short vss;
volatile vector signed int vsi;
volatile vector signed long long vsl;
volatile vector unsigned char vuc;
volatile vector unsigned short vus;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector bool char vbc;
volatile vector bool short vbs;
volatile vector bool int vbi;
volatile vector bool long long vbl;
volatile vector float vf;
volatile vector double vd;

volatile unsigned int len;

void test_nnp_assist(void) {
  vf = vec_extend_to_fp32_hi(vus, -1);         // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vf = vec_extend_to_fp32_hi(vus, 16);         // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vf = vec_extend_to_fp32_hi(vus, len);        // expected-error {{argument to '__builtin_s390_vclfnhs' must be a constant integer}}

  vf = vec_extend_to_fp32_lo(vus, -1);         // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vf = vec_extend_to_fp32_lo(vus, 16);         // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vf = vec_extend_to_fp32_lo(vus, len);        // expected-error {{argument to '__builtin_s390_vclfnls' must be a constant integer}}

  vus = vec_round_from_fp32(vf, vf, -1);       // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vus = vec_round_from_fp32(vf, vf, 16);       // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vus = vec_round_from_fp32(vf, vf, len);      // expected-error {{argument to '__builtin_s390_vcrnfs' must be a constant integer}}

  vus = vec_convert_to_fp16(vus, -1);          // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vus = vec_convert_to_fp16(vus, 16);          // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vus = vec_convert_to_fp16(vus, len);         // expected-error {{argument to '__builtin_s390_vcfn' must be a constant integer}}

  vus = vec_convert_from_fp16(vus, -1);        // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vus = vec_convert_from_fp16(vus, 16);        // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vus = vec_convert_from_fp16(vus, len);       // expected-error {{argument to '__builtin_s390_vcnf' must be a constant integer}}
}
