// RUN: %clang_cc1 -fsyntax-only -verify=scalar,neon -std=c++11 \
// RUN:   -triple aarch64 -target-cpu cortex-a75 \
// RUN:   -target-feature +bf16 -target-feature +neon -Wno-unused %s
// RUN: %clang_cc1 -fsyntax-only -verify=scalar,neon -std=c++11 \
// RUN:   -triple arm-arm-none-eabi -target-cpu cortex-a53 \
// RUN:   -target-feature +bf16 -target-feature +neon -Wno-unused %s

// The types should be available under AArch64 even without the bf16 feature
// RUN: %clang_cc1 -fsyntax-only -verify=scalar -DNONEON -std=c++11 \
// RUN:   -triple aarch64 -target-cpu cortex-a75 \
// RUN:   -target-feature -bf16 -target-feature +neon -Wno-unused %s

// REQUIRES: aarch64-registered-target || arm-registered-target

void test(bool b) {
  __bf16 bf16;

  bf16 + bf16;
  bf16 - bf16;
  bf16 * bf16;
  bf16 / bf16;
  ++bf16;
  --bf16;

  __fp16 fp16;

  bf16 + fp16;
  fp16 + bf16;
  bf16 - fp16;
  fp16 - bf16;
  bf16 * fp16;
  fp16 * bf16;
  bf16 / fp16;
  fp16 / bf16;
  bf16 = fp16; // scalar-error {{assigning to '__bf16' from incompatible type '__fp16'}}
  fp16 = bf16; // scalar-error {{assigning to '__fp16' from incompatible type '__bf16'}}
  bf16 + (b ? fp16 : bf16);
}

#ifndef NONEON

#include <arm_neon.h>

void test_vector(bfloat16x4_t a, bfloat16x4_t b, float16x4_t c) {
  a + b;
  a - b;
  a * b;
  a / b;

  a + c;
  a - c;
  a * c;
  a / c;
  c + b;
  c - b;
  c * b;
  c / b;
}
#endif