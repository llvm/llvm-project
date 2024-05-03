// RUN: %clang_cc1 -triple aarch64-none-linux-gnu  -target-feature +sme2p1 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target
#include "arm_sme.h"

svuint8x2_t  test_sme2p1(svuint8x2_t  x) __arm_streaming {
  // expected-no-diagnostics
  return x;
}

