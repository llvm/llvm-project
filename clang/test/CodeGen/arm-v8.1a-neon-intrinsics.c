// RUN: %clang_cc1 -triple armv8.1a-linux-gnu -target-feature +neon \
// RUN:  -O3 -S -o - %s \
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon \
// RUN:  -target-feature +v8.1a -O3 -S -o - %s \
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AARCH64

#include <arm_neon.h>

// CHECK-LABEL: test_vqrdmlah_s16
int16x4_t test_vqrdmlah_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: vqrdmlah.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  return vqrdmlah_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlah_s32
int32x2_t test_vqrdmlah_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: vqrdmlah.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  return vqrdmlah_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlahq_s16
int16x8_t test_vqrdmlahq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
// CHECK-ARM: vqrdmlah.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  return vqrdmlahq_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlahq_s32
int32x4_t test_vqrdmlahq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
// CHECK-ARM: vqrdmlah.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  return vqrdmlahq_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlah_lane_s16
int16x4_t test_vqrdmlah_lane_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: vqrdmlah.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[3]
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
  return vqrdmlah_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlah_lane_s32
int32x2_t test_vqrdmlah_lane_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: vqrdmlah.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[1]
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  return vqrdmlah_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlahq_lane_s16
int16x8_t test_vqrdmlahq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t c) {
// CHECK-ARM: vqrdmlah.s16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[3]
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
  return vqrdmlahq_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlahq_lane_s32
int32x4_t test_vqrdmlahq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t c) {
// CHECK-ARM: vqrdmlah.s32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[1]
// CHECK-AARCH64: sqrdmlah {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  return vqrdmlahq_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlsh_s16
int16x4_t test_vqrdmlsh_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: vqrdmlsh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  return vqrdmlsh_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlsh_s32
int32x2_t test_vqrdmlsh_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: vqrdmlsh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  return vqrdmlsh_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlshq_s16
int16x8_t test_vqrdmlshq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
// CHECK-ARM: vqrdmlsh.s16 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  return vqrdmlshq_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlshq_s32
int32x4_t test_vqrdmlshq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
// CHECK-ARM: vqrdmlsh.s32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  return vqrdmlshq_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlsh_lane_s16
int16x4_t test_vqrdmlsh_lane_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: vqrdmlsh.s16 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[3]
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
  return vqrdmlsh_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlsh_lane_s32
int32x2_t test_vqrdmlsh_lane_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: vqrdmlsh.s32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}[1]
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  return vqrdmlsh_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlshq_lane_s16
int16x8_t test_vqrdmlshq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t c) {
// CHECK-ARM: vqrdmlsh.s16 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[3]
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
  return vqrdmlshq_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlshq_lane_s32
int32x4_t test_vqrdmlshq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t c) {
// CHECK-ARM: vqrdmlsh.s32 q{{[0-9]+}}, q{{[0-9]+}}, d{{[0-9]+}}[1]
// CHECK-AARCH64: sqrdmlsh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  return vqrdmlshq_lane_s32(a, b, c, 1);
}

