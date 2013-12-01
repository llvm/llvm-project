// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

float32_t test_vcvtxd_f32_f64(float64_t a) {
// CHECK: test_vcvtxd_f32_f64
// CHECK: fcvtxn {{s[0-9]+}}, {{d[0-9]+}}
  return (float32_t)vcvtxd_f32_f64(a);
}

int32_t test_vcvtas_s32_f32(float32_t a) {
// CHECK: test_vcvtas_s32_f32
// CHECK: fcvtas {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vcvtas_s32_f32(a);
}

int64_t test_test_vcvtad_s64_f64(float64_t a) {
// CHECK: test_test_vcvtad_s64_f64
// CHECK: fcvtas {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vcvtad_s64_f64(a);
}

uint32_t test_vcvtas_u32_f32(float32_t a) {
// CHECK: test_vcvtas_u32_f32
// CHECK: fcvtau {{s[0-9]+}}, {{s[0-9]+}}
  return (uint32_t)vcvtas_u32_f32(a);
}

uint64_t test_vcvtad_u64_f64(float64_t a) {
// CHECK: test_vcvtad_u64_f64
// CHECK: fcvtau {{d[0-9]+}}, {{d[0-9]+}}
  return (uint64_t)vcvtad_u64_f64(a);
}

int32_t test_vcvtms_s32_f32(float32_t a) {
// CHECK: test_vcvtms_s32_f32
// CHECK: fcvtms {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vcvtms_s32_f32(a);
}

int64_t test_vcvtmd_s64_f64(float64_t a) {
// CHECK: test_vcvtmd_s64_f64
// CHECK: fcvtms {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vcvtmd_s64_f64(a);
}

uint32_t test_vcvtms_u32_f32(float32_t a) {
// CHECK: test_vcvtms_u32_f32
// CHECK: fcvtmu {{s[0-9]+}}, {{s[0-9]+}}
  return (uint32_t)vcvtms_u32_f32(a);
}

uint64_t test_vcvtmd_u64_f64(float64_t a) {
// CHECK: test_vcvtmd_u64_f64
// CHECK: fcvtmu {{d[0-9]+}}, {{d[0-9]+}}
  return (uint64_t)vcvtmd_u64_f64(a);
}

int32_t test_vcvtns_s32_f32(float32_t a) {
// CHECK: test_vcvtns_s32_f32
// CHECK: fcvtns {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vcvtns_s32_f32(a);
}

int64_t test_vcvtnd_s64_f64(float64_t a) {
// CHECK: test_vcvtnd_s64_f64
// CHECK: fcvtns {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vcvtnd_s64_f64(a);
}

uint32_t test_vcvtns_u32_f32(float32_t a) {
// CHECK: test_vcvtns_u32_f32
// CHECK: fcvtnu {{s[0-9]+}}, {{s[0-9]+}}
  return (uint32_t)vcvtns_u32_f32(a);
}

uint64_t test_vcvtnd_u64_f64(float64_t a) {
// CHECK: test_vcvtnd_u64_f64
// CHECK: fcvtnu {{d[0-9]+}}, {{d[0-9]+}}
  return (uint64_t)vcvtnd_u64_f64(a);
}

int32_t test_vcvtps_s32_f32(float32_t a) {
// CHECK: test_vcvtps_s32_f32
// CHECK: fcvtps {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vcvtps_s32_f32(a);
}

int64_t test_vcvtpd_s64_f64(float64_t a) {
// CHECK: test_vcvtpd_s64_f64
// CHECK: fcvtps {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vcvtpd_s64_f64(a);
}

uint32_t test_vcvtps_u32_f32(float32_t a) {
// CHECK: test_vcvtps_u32_f32
// CHECK: fcvtpu {{s[0-9]+}}, {{s[0-9]+}}
  return (uint32_t)vcvtps_u32_f32(a);
}

uint64_t test_vcvtpd_u64_f64(float64_t a) {
// CHECK: test_vcvtpd_u64_f64
// CHECK: fcvtpu {{d[0-9]+}}, {{d[0-9]+}}
  return (uint64_t)vcvtpd_u64_f64(a);
}

int32_t test_vcvts_s32_f32(float32_t a) {
// CHECK: test_vcvts_s32_f32
// CHECK: fcvtzs {{s[0-9]+}}, {{s[0-9]+}}
  return (int32_t)vcvts_s32_f32(a);
}

int64_t test_vcvtd_s64_f64(float64_t a) {
// CHECK: test_vcvtd_s64_f64
// CHECK: fcvtzs {{d[0-9]+}}, {{d[0-9]+}}
  return (int64_t)vcvtd_s64_f64(a);
}

uint32_t test_vcvts_u32_f32(float32_t a) {
// CHECK: test_vcvts_u32_f32
// CHECK: fcvtzu {{s[0-9]+}}, {{s[0-9]+}}
  return (uint32_t)vcvts_u32_f32(a);
}

uint64_t test_vcvtd_u64_f64(float64_t a) {
// CHECK: test_vcvtd_u64_f64
// CHECK: fcvtzu {{d[0-9]+}}, {{d[0-9]+}}
  return (uint64_t)vcvtd_u64_f64(a);
}
