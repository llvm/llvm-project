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
