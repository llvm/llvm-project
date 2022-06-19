// RUN: %clang_cc1 -triple armebv7-linux-gnueabihf -target-feature +neon \
// RUN:  -Wdeclaration-after-statement -fsyntax-only -verify %s
// REQUIRES: arm-registered-target
// https://github.com/llvm/llvm-project/issues/54062
#include <arm_neon.h>

uint8x16_t a;

uint8x16_t x(void) {
  return vshrq_n_u8(a, 8);
}
// expected-no-diagnostics
