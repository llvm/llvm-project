// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 \
// RUN: -triple aarch64-arm-none-eabi -target-cpu cortex-a75 \
// RUN: -target-feature +bf16 -target-feature +neon %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 \
// RUN: -triple arm-arm-none-eabi -target-cpu cortex-a53 \
// RUN: -target-feature +bf16 -target-feature +neon %s

void test(bool b) {
  __bf16 bf16;

  bf16 + bf16; // expected-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}
  bf16 - bf16; // expected-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}
  bf16 * bf16; // expected-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}
  bf16 / bf16; // expected-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}

  __fp16 fp16;

  bf16 + fp16; // expected-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 + bf16; // expected-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 - fp16; // expected-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 - bf16; // expected-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 * fp16; // expected-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 * bf16; // expected-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 / fp16; // expected-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 / bf16; // expected-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 = fp16; // expected-error {{assigning to '__bf16' from incompatible type '__fp16'}}
  fp16 = bf16; // expected-error {{assigning to '__fp16' from incompatible type '__bf16'}}
  bf16 + (b ? fp16 : bf16); // expected-error {{incompatible operand types ('__fp16' and '__bf16')}}
}

#include <arm_neon.h>

void test_vector(bfloat16x4_t a, bfloat16x4_t b, float16x4_t c) {
  a + b; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'bfloat16x4_t')}}
  a - b; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'bfloat16x4_t')}}
  a * b; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'bfloat16x4_t')}}
  a / b; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'bfloat16x4_t')}}

  a + c; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'float16x4_t' (vector of 4 'float16_t' values))}}
  a - c; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'float16x4_t' (vector of 4 'float16_t' values))}}
  a * c; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'float16x4_t' (vector of 4 'float16_t' values))}}
  a / c; // expected-error {{invalid operands to binary expression ('bfloat16x4_t' (vector of 4 'bfloat16_t' values) and 'float16x4_t' (vector of 4 'float16_t' values))}}
  c + b; // expected-error {{invalid operands to binary expression ('float16x4_t' (vector of 4 'float16_t' values) and 'bfloat16x4_t' (vector of 4 'bfloat16_t' values))}}
  c - b; // expected-error {{invalid operands to binary expression ('float16x4_t' (vector of 4 'float16_t' values) and 'bfloat16x4_t' (vector of 4 'bfloat16_t' values))}}
  c * b; // expected-error {{invalid operands to binary expression ('float16x4_t' (vector of 4 'float16_t' values) and 'bfloat16x4_t' (vector of 4 'bfloat16_t' values))}}
  c / b; // expected-error {{invalid operands to binary expression ('float16x4_t' (vector of 4 'float16_t' values) and 'bfloat16x4_t' (vector of 4 'bfloat16_t' values))}}
}
