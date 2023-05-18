// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-none-netbsd-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test whether arm_neon.h works as expected in C++.

#include "arm_neon.h"

poly64x1_t test_vld1_p64(poly64_t const * ptr) {
  // CHECK: test_vld1_p64
  return vld1_p64(ptr);
  // CHECK:  {{ld1 { v[0-9]+.1d }|ldr d[0-9]+}}, [{{x[0-9]+|sp}}]
}
