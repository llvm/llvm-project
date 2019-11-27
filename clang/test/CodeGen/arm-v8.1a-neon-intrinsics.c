// RUN: %clang_cc1 -triple armv8.1a-linux-gnu -target-abi apcs-gnu -target-feature +neon \
// RUN:  -S -emit-llvm -o - %s \
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon \
// RUN:  -target-feature +v8.1a -S -emit-llvm -o - %s \
// RUN:  | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AARCH64

// REQUIRES: arm-registered-target,aarch64-registered-target

#include <arm_neon.h>

// CHECK-LABEL: test_vqrdmlah_s16
int16x4_t test_vqrdmlah_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-ARM: call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})

// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
  return vqrdmlah_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlah_s32
int32x2_t test_vqrdmlah_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-ARM: call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})

// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
  return vqrdmlah_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlahq_s16
int16x8_t test_vqrdmlahq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
// CHECK-ARM: call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-ARM: call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})

// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
  return vqrdmlahq_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlahq_s32
int32x4_t test_vqrdmlahq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
// CHECK-ARM: call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-ARM: call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})

// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
  return vqrdmlahq_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlah_lane_s16
int16x4_t test_vqrdmlah_lane_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK-ARM: call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-ARM: call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})

// CHECK-AARCH64: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
  return vqrdmlah_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlah_lane_s32
int32x2_t test_vqrdmlah_lane_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> <i32 1, i32 1>
// CHECK-ARM: call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-ARM: call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})

// CHECK-AARCH64: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> <i32 1, i32 1>
// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
  return vqrdmlah_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlahq_lane_s16
int16x8_t test_vqrdmlahq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t c) {
// CHECK-ARM: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK-ARM: call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-ARM: call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})

// CHECK-AARCH64: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
  return vqrdmlahq_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlahq_lane_s32
int32x4_t test_vqrdmlahq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t c) {
// CHECK-ARM: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK-ARM: call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-ARM: call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})

// CHECK-AARCH64: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
  return vqrdmlahq_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlsh_s16
int16x4_t test_vqrdmlsh_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-ARM: call <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})

// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
  return vqrdmlsh_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlsh_s32
int32x2_t test_vqrdmlsh_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-ARM: call <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})

// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
  return vqrdmlsh_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlshq_s16
int16x8_t test_vqrdmlshq_s16(int16x8_t a, int16x8_t b, int16x8_t c) {
// CHECK-ARM: call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-ARM: call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})

// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
  return vqrdmlshq_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlshq_s32
int32x4_t test_vqrdmlshq_s32(int32x4_t a, int32x4_t b, int32x4_t c) {
// CHECK-ARM: call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-ARM: call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})

// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
  return vqrdmlshq_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlsh_lane_s16
int16x4_t test_vqrdmlsh_lane_s16(int16x4_t a, int16x4_t b, int16x4_t c) {
// CHECK-ARM: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK-ARM: call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-ARM: call <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})

// CHECK-AARCH64: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK-AARCH64: call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
  return vqrdmlsh_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlsh_lane_s32
int32x2_t test_vqrdmlsh_lane_s32(int32x2_t a, int32x2_t b, int32x2_t c) {
// CHECK-ARM: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> <i32 1, i32 1>
// CHECK-ARM: call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-ARM: call <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})

// CHECK-AARCH64: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> <i32 1, i32 1>
// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
  return vqrdmlsh_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlshq_lane_s16
int16x8_t test_vqrdmlshq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t c) {
// CHECK-ARM: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK-ARM: call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-ARM: call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})

// CHECK-AARCH64: shufflevector <4 x i16> {{%.*}}, <4 x i16> {{%.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK-AARCH64: call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
  return vqrdmlshq_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlshq_lane_s32
int32x4_t test_vqrdmlshq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t c) {
// CHECK-ARM: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK-ARM: call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-ARM: call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})

// CHECK-AARCH64: shufflevector <2 x i32> {{%.*}}, <2 x i32> {{%.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK-AARCH64: call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
  return vqrdmlshq_lane_s32(a, b, c, 1);
}
