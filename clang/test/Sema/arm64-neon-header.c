// RUN: %clang_cc1 -triple arm64-apple-darwin -target-feature +neon -Wvector-conversion -fsyntax-only -ffreestanding -verify %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

int16x8_t foo(uint8x8_t p0, int16x8_t p1) {
  return vqmovun_high_s16(p0, p1); // expected-warning {{incompatible vector types returning 'uint8x16_t'}}
}
