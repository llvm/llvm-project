// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon \
// RUN:  -target-feature +crypto -S -emit-llvm -o - %s \
// RUN:  | FileCheck %s

// RUN: not %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon \
// RUN: -S -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=CHECK-NO-CRYPTO %s

//The front-end requires the addition of both +crypto and +sm4 in the 
// command line, however the back-end requires only +sm4 (includes sm4&sm3)

#include <arm_neon.h>

void test_vsm3partw1(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3partw1(
  // CHECK-NO-CRYPTO: warning: implicit declaration of function 'vsm3partw1q_u32' is invalid in C99
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3partw1
  uint32x4_t result = vsm3partw1q_u32(a, b, c);
}

void test_vsm3partw2(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3partw2(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3partw2
  uint32x4_t result = vsm3partw2q_u32(a, b, c);
}

void test_vsm3ss1(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3ss1(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3ss1
  uint32x4_t result = vsm3ss1q_u32(a, b, c);
}

void test_vsm3tt1a(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3tt1a(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3tt1a
  uint32x4_t result = vsm3tt1aq_u32(a, b, c, 2);
}

void test_vsm3tt1b(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3tt1b(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3tt1b
  uint32x4_t result = vsm3tt1bq_u32(a, b, c, 2);
}

void test_vsm3tt2a(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3tt2a(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3tt2a
  uint32x4_t result = vsm3tt2aq_u32(a, b, c, 2);
}

void test_vsm3tt2b(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  // CHECK-LABEL: @test_vsm3tt2b(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm3tt2b
  uint32x4_t result = vsm3tt2bq_u32(a, b, c, 2);
}

void test_vsm4e(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: @test_vsm4e(
  // CHECK: call <4 x i32> @llvm.aarch64.crypto.sm4e
  uint32x4_t result = vsm4eq_u32(a, b);
}

void test_vsm4ekey(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: @test_vsm4ekey(
  // CHECK: call  <4 x i32> @llvm.aarch64.crypto.sm4ekey
  uint32x4_t result = vsm4ekeyq_u32(a, b);
}
