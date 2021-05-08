// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -target-feature +neon \
// RUN:  -target-feature +sha2 -target-feature +aes \
// RUN:  -target-cpu cortex-a57 -emit-llvm -O1 -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -target-feature +sha2 -target-feature +aes \
// RUN:   -emit-llvm -O1 -o - %s | FileCheck %s
// RUN: not %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -S -O3 -o - %s 2>&1 | FileCheck --check-prefix=CHECK-NO-CRYPTO %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

uint8x16_t test_vaeseq_u8(uint8x16_t data, uint8x16_t key) {
  // CHECK-LABEL: @test_vaeseq_u8
  // CHECK-NO-CRYPTO: warning: implicit declaration of function 'vaeseq_u8' is invalid in C99
  return vaeseq_u8(data, key);
  // CHECK: call <16 x i8> @llvm.{{arm.neon|aarch64.crypto}}.aese(<16 x i8> %data, <16 x i8> %key)
}

uint8x16_t test_vaesdq_u8(uint8x16_t data, uint8x16_t key) {
  // CHECK-LABEL: @test_vaesdq_u8
  return vaesdq_u8(data, key);
  // CHECK: call <16 x i8> @llvm.{{arm.neon|aarch64.crypto}}.aesd(<16 x i8> %data, <16 x i8> %key)
}

uint8x16_t test_vaesmcq_u8(uint8x16_t data) {
  // CHECK-LABEL: @test_vaesmcq_u8
  return vaesmcq_u8(data);
  // CHECK: call <16 x i8> @llvm.{{arm.neon|aarch64.crypto}}.aesmc(<16 x i8> %data)
}

uint8x16_t test_vaesimcq_u8(uint8x16_t data) {
  // CHECK-LABEL: @test_vaesimcq_u8
  return vaesimcq_u8(data);
  // CHECK: call <16 x i8> @llvm.{{arm.neon|aarch64.crypto}}.aesimc(<16 x i8> %data)
}

uint32_t test_vsha1h_u32(uint32_t hash_e) {
  // CHECK-LABEL: @test_vsha1h_u32
  return vsha1h_u32(hash_e);
  // CHECK: call i32 @llvm.{{arm.neon|aarch64.crypto}}.sha1h(i32 %hash_e)
}

uint32x4_t test_vsha1su1q_u32(uint32x4_t w0_3, uint32x4_t w12_15) {
  // CHECK-LABEL: @test_vsha1su1q_u32
  return vsha1su1q_u32(w0_3, w12_15);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha1su1(<4 x i32> %w0_3, <4 x i32> %w12_15)
}

uint32x4_t test_vsha256su0q_u32(uint32x4_t w0_3, uint32x4_t w4_7) {
  // CHECK-LABEL: @test_vsha256su0q_u32
  return vsha256su0q_u32(w0_3, w4_7);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha256su0(<4 x i32> %w0_3, <4 x i32> %w4_7)
}

uint32x4_t test_vsha1cq_u32(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK-LABEL: @test_vsha1cq_u32
  return vsha1cq_u32(hash_abcd, hash_e, wk);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha1c(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
}

uint32x4_t test_vsha1pq_u32(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK-LABEL: @test_vsha1pq_u32
  return vsha1pq_u32(hash_abcd, hash_e, wk);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha1p(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
}

uint32x4_t test_vsha1mq_u32(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK-LABEL: @test_vsha1mq_u32
  return vsha1mq_u32(hash_abcd, hash_e, wk);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha1m(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk)
}

uint32x4_t test_vsha1su0q_u32(uint32x4_t w0_3, uint32x4_t w4_7, uint32x4_t w8_11) {
  // CHECK-LABEL: @test_vsha1su0q_u32
  return vsha1su0q_u32(w0_3, w4_7, w8_11);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha1su0(<4 x i32> %w0_3, <4 x i32> %w4_7, <4 x i32> %w8_11)
}

uint32x4_t test_vsha256hq_u32(uint32x4_t hash_abcd, uint32x4_t hash_efgh, uint32x4_t wk) {
  // CHECK-LABEL: @test_vsha256hq_u32
  return vsha256hq_u32(hash_abcd, hash_efgh, wk);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha256h(<4 x i32> %hash_abcd, <4 x i32> %hash_efgh, <4 x i32> %wk)
}

uint32x4_t test_vsha256h2q_u32(uint32x4_t hash_efgh, uint32x4_t hash_abcd, uint32x4_t wk) {
  // CHECK-LABEL: @test_vsha256h2q_u32
  return vsha256h2q_u32(hash_efgh, hash_abcd, wk);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha256h2(<4 x i32> %hash_efgh, <4 x i32> %hash_abcd, <4 x i32> %wk)
}

uint32x4_t test_vsha256su1q_u32(uint32x4_t w0_3, uint32x4_t w8_11, uint32x4_t w12_15) {
  // CHECK-LABEL: @test_vsha256su1q_u32
  return vsha256su1q_u32(w0_3, w8_11, w12_15);
  // CHECK: call <4 x i32> @llvm.{{arm.neon|aarch64.crypto}}.sha256su1(<4 x i32> %w0_3, <4 x i32> %w8_11, <4 x i32> %w12_15)
}
