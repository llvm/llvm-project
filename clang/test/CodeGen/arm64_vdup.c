// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD duplicate lane and n intrinsics

#include <arm_neon.h>

void test_vdup_lane_s64(int64x1_t a1) {
  // CHECK-LABEL: test_vdup_lane_s64
  vdup_lane_s64(a1, 0);
  // CHECK: shufflevector
}

void test_vdup_lane_u64(uint64x1_t a1) {
  // CHECK-LABEL: test_vdup_lane_u64
  vdup_lane_u64(a1, 0);
  // CHECK: shufflevector
}

// uncomment out the following code once scalar_to_vector in the backend
// works (for 64 bit?).  Change the "CHECK@" to "CHECK<colon>"
/*
float64x1_t test_vdup_n_f64(float64_t a1) {
  // CHECK-LABEL@ test_vdup_n_f64
  return vdup_n_f64(a1);
  // match that an element is inserted into part 0
  // CHECK@ insertelement {{.*, i32 0 *$}}
}
*/

float16x8_t test_vdupq_n_f16(float16_t *a1) {
  // CHECK-LABEL: test_vdupq_n_f16
  return vdupq_n_f16(*a1);
  // match that an element is inserted into parts 0-7.  The backend better
  // turn that into a single dup intruction
  // CHECK: insertelement {{.*, i32 0 *$}}
  // CHECK: insertelement {{.*, i32 1 *$}}
  // CHECK: insertelement {{.*, i32 2 *$}}
  // CHECK: insertelement {{.*, i32 3 *$}}
  // CHECK: insertelement {{.*, i32 4 *$}}
  // CHECK: insertelement {{.*, i32 5 *$}}
  // CHECK: insertelement {{.*, i32 6 *$}}
  // CHECK: insertelement {{.*, i32 7 *$}}
}
