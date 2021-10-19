// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s | FileCheck %s

#include <arm_sve.h>

svbool_t test_svdupq_n_b8_const()
{
  // CHECK-LABEL: test_svdupq_n_b8_const
  // CHECK: ptrue p0.h
  // CHECK-NEXT: ret
  return svdupq_n_b8(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
}

svbool_t test_svdupq_n_b16_const()
{
  // CHECK-LABEL: test_svdupq_n_b16_const
  // CHECK: ptrue p0.h
  // CHECK-NEXT: ret
  return svdupq_n_b16(1, 1, 1, 1, 1, 1, 1, 1);
}

svbool_t test_svdupq_n_b32_const()
{
  // CHECK-LABEL: test_svdupq_n_b32_const
  // CHECK: ptrue p0.s
  // CHECK-NEXT: ret
  return svdupq_n_b32(1, 1, 1, 1);
}

svbool_t test_svdupq_n_b64_const()
{
  // CHECK-LABEL: test_svdupq_n_b64_const
  // CHECK: ptrue p0.d
  // CHECK-NEXT: ret
  return svdupq_n_b64(1, 1);
}
