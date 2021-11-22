// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-feature +neon -target-feature +aes -target-feature +sha2 -ffreestanding -Os -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-feature +neon -target-feature +aes -target-feature +sha2 -ffreestanding -fexperimental-new-pass-manager -Os -S -o - %s | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

uint8x16_t test_aese(uint8x16_t data, uint8x16_t key) {
  // CHECK-LABEL: test_aese:
  // CHECK: aese.16b v0, v1
  return vaeseq_u8(data, key);
}

uint8x16_t test_aesd(uint8x16_t data, uint8x16_t key) {
  // CHECK-LABEL: test_aesd:
  // CHECK: aesd.16b v0, v1
  return vaesdq_u8(data, key);
}

uint8x16_t test_aesmc(uint8x16_t data, uint8x16_t key) {
  // CHECK-LABEL: test_aesmc:
  // CHECK: aesmc.16b v0, v0
  return vaesmcq_u8(data);
}

uint8x16_t test_aesimc(uint8x16_t data, uint8x16_t key) {
  // CHECK-LABEL: test_aesimc:
  // CHECK: aesimc.16b v0, v0
  return vaesimcq_u8(data);
}

uint32x4_t test_sha1c(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK-LABEL: test_sha1c:
  // CHECK: fmov [[HASH_E:s[0-9]+]], w0
  // CHECK: sha1c.4s q0, [[HASH_E]], v1
  return vsha1cq_u32(hash_abcd, hash_e, wk);
}

uint32x4_t test_sha1p(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK-LABEL: test_sha1p:
  // CHECK: fmov [[HASH_E:s[0-9]+]], w0
  // CHECK: sha1p.4s q0, [[HASH_E]], v1
  return vsha1pq_u32(hash_abcd, hash_e, wk);
}

uint32x4_t test_sha1m(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK-LABEL: test_sha1m:
  // CHECK: fmov [[HASH_E:s[0-9]+]], w0
  // CHECK: sha1m.4s q0, [[HASH_E]], v1
  return vsha1mq_u32(hash_abcd, hash_e, wk);
}

uint32_t test_sha1h(uint32_t hash_e) {
  // CHECK-LABEL: test_sha1h:
  // CHECK: fmov [[HASH_E:s[0-9]+]], w0
  // CHECK: sha1h [[RES:s[0-9]+]], [[HASH_E]]
  // CHECK: fmov w0, [[RES]]
  return vsha1h_u32(hash_e);
}

uint32x4_t test_sha1su0(uint32x4_t wk0_3, uint32x4_t wk4_7, uint32x4_t wk8_11) {
  // CHECK-LABEL: test_sha1su0:
  // CHECK: sha1su0.4s v0, v1, v2
  return vsha1su0q_u32(wk0_3, wk4_7, wk8_11);
}

uint32x4_t test_sha1su1(uint32x4_t wk0_3, uint32x4_t wk12_15) {
  // CHECK-LABEL: test_sha1su1:
  // CHECK: sha1su1.4s v0, v1
  return vsha1su1q_u32(wk0_3, wk12_15);
}

uint32x4_t test_sha256h(uint32x4_t hash_abcd, uint32x4_t hash_efgh, uint32x4_t wk) {
  // CHECK-LABEL: test_sha256h:
  // CHECK: sha256h.4s q0, q1, v2
  return vsha256hq_u32(hash_abcd, hash_efgh, wk);
}

uint32x4_t test_sha256h2(uint32x4_t hash_efgh, uint32x4_t hash_abcd, uint32x4_t wk) {
  // CHECK-LABEL: test_sha256h2:
  // CHECK: sha256h2.4s q0, q1, v2
  return vsha256h2q_u32(hash_efgh, hash_abcd, wk);
}

uint32x4_t test_sha256su0(uint32x4_t w0_3, uint32x4_t w4_7) {
  // CHECK-LABEL: test_sha256su0:
  // CHECK: sha256su0.4s v0, v1
  return vsha256su0q_u32(w0_3, w4_7);
}

uint32x4_t test_sha256su1(uint32x4_t w0_3, uint32x4_t w8_11, uint32x4_t w12_15) {
  // CHECK-LABEL: test_sha256su1:
  // CHECK: sha256su1.4s v0, v1, v2
  return vsha256su1q_u32(w0_3, w8_11, w12_15);
}
