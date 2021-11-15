// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon \
// RUN:  -target-feature +v8.1a -S -emit-llvm -o - %s | FileCheck %s

// REQUIRES: aarch64-registered-target

 #include <arm_neon.h>

// CHECK-LABEL: test_vqrdmlah_laneq_s16
int16x4_t test_vqrdmlah_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
// CHECK: shufflevector <8 x i16> {{%.*}}, <8 x i16> {{%.*}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK: call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK: call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
  return vqrdmlah_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: test_vqrdmlah_laneq_s32
int32x2_t test_vqrdmlah_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
// CHECK: shufflevector <4 x i32> {{%.*}}, <4 x i32> {{%.*}}, <2 x i32> <i32 3, i32 3>
// CHECK: call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK: call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
  return vqrdmlah_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: test_vqrdmlahq_laneq_s16
int16x8_t test_vqrdmlahq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
// CHECK: shufflevector <8 x i16> {{%.*}}, <8 x i16> {{%.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK: call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
  return vqrdmlahq_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: test_vqrdmlahq_laneq_s32
int32x4_t test_vqrdmlahq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
// CHECK: shufflevector <4 x i32> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK: call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
  return vqrdmlahq_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: test_vqrdmlahh_s16
int16_t test_vqrdmlahh_s16(int16_t a, int16_t b, int16_t c) {
// CHECK: [[insb:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insc:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[mul:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[insb]], <4 x i16> [[insc]])
// CHECK: extractelement <4 x i16> [[mul]], i64 0
// CHECK: [[insa:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insmul:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[add:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> [[insa]], <4 x i16> [[insmul]])
// CHECK: extractelement <4 x i16> [[add]], i64 0
  return vqrdmlahh_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlahs_s32
int32_t test_vqrdmlahs_s32(int32_t a, int32_t b, int32_t c) {
// CHECK: call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// CHECK: call i32 @llvm.aarch64.neon.sqadd.i32(i32 {{%.*}}, i32 {{%.*}})
  return vqrdmlahs_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlahh_lane_s16
int16_t test_vqrdmlahh_lane_s16(int16_t a, int16_t b, int16x4_t c) {
// CHECK: extractelement <4 x i16> {{%.*}}, i32 3
// CHECK: [[insb:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insc:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[mul:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[insb]], <4 x i16> [[insc]])
// CHECK: extractelement <4 x i16> [[mul]], i64 0
// CHECK: [[insa:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insmul:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[add:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> [[insa]], <4 x i16> [[insmul]])
// CHECK: extractelement <4 x i16> [[add]], i64 0
  return vqrdmlahh_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlahs_lane_s32
int32_t test_vqrdmlahs_lane_s32(int32_t a, int32_t b, int32x2_t c) {
// CHECK: extractelement <2 x i32> {{%.*}}, i32 1
// CHECK: call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// CHECK: call i32 @llvm.aarch64.neon.sqadd.i32(i32 {{%.*}}, i32 {{%.*}})
  return vqrdmlahs_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlahh_laneq_s16
int16_t test_vqrdmlahh_laneq_s16(int16_t a, int16_t b, int16x8_t c) {
// CHECK: extractelement <8 x i16> {{%.*}}, i32 7
// CHECK: [[insb:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insc:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[mul:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[insb]], <4 x i16> [[insc]])
// CHECK: extractelement <4 x i16> [[mul]], i64 0
// CHECK: [[insa:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insmul:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[add:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> [[insa]], <4 x i16> [[insmul]])
// CHECK: extractelement <4 x i16> [[add]], i64 0
  return vqrdmlahh_laneq_s16(a, b, c, 7);
}

// CHECK-LABEL: test_vqrdmlahs_laneq_s32
int32_t test_vqrdmlahs_laneq_s32(int32_t a, int32_t b, int32x4_t c) {
// CHECK: extractelement <4 x i32> {{%.*}}, i32 3
// CHECK: call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// CHECK: call i32 @llvm.aarch64.neon.sqadd.i32(i32 {{%.*}}, i32 {{%.*}})
  return vqrdmlahs_laneq_s32(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlsh_laneq_s16
int16x4_t test_vqrdmlsh_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
// CHECK: shufflevector <8 x i16> {{%.*}}, <8 x i16> {{%.*}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK: call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
// CHECK: call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> {{%.*}}, <4 x i16> {{%.*}})
  return vqrdmlsh_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: test_vqrdmlsh_laneq_s32
int32x2_t test_vqrdmlsh_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
// CHECK: shufflevector <4 x i32> {{%.*}}, <4 x i32> {{%.*}}, <2 x i32> <i32 3, i32 3>
// CHECK: call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
// CHECK: call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> {{%.*}}, <2 x i32> {{%.*}})
  return vqrdmlsh_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: test_vqrdmlshq_laneq_s16
int16x8_t test_vqrdmlshq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
// CHECK: shufflevector <8 x i16> {{%.*}}, <8 x i16> {{%.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
// CHECK: call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> {{%.*}}, <8 x i16> {{%.*}})
  return vqrdmlshq_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: test_vqrdmlshq_laneq_s32
int32x4_t test_vqrdmlshq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
// CHECK: shufflevector <4 x i32> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
// CHECK: call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> {{%.*}}, <4 x i32> {{%.*}})
  return vqrdmlshq_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: test_vqrdmlshh_s16
int16_t test_vqrdmlshh_s16(int16_t a, int16_t b, int16_t c) {
// CHECK: [[insb:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insc:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[mul:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[insb]], <4 x i16> [[insc]])
// CHECK: extractelement <4 x i16> [[mul]], i64 0
// CHECK: [[insa:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insmul:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[sub:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> [[insa]], <4 x i16> [[insmul]])
// CHECK: extractelement <4 x i16> [[sub]], i64 0
  return vqrdmlshh_s16(a, b, c);
}

// CHECK-LABEL: test_vqrdmlshs_s32
int32_t test_vqrdmlshs_s32(int32_t a, int32_t b, int32_t c) {
// CHECK: call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// CHECK: call i32 @llvm.aarch64.neon.sqsub.i32(i32 {{%.*}}, i32 {{%.*}})
  return vqrdmlshs_s32(a, b, c);
}

// CHECK-LABEL: test_vqrdmlshh_lane_s16
int16_t test_vqrdmlshh_lane_s16(int16_t a, int16_t b, int16x4_t c) {
// CHECK: extractelement <4 x i16> {{%.*}}, i32 3
// CHECK: [[insb:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insc:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[mul:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[insb]], <4 x i16> [[insc]])
// CHECK: extractelement <4 x i16> [[mul]], i64 0
// CHECK: [[insa:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insmul:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[sub:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> [[insa]], <4 x i16> [[insmul]])
// CHECK: extractelement <4 x i16> [[sub]], i64 0
  return vqrdmlshh_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: test_vqrdmlshs_lane_s32
int32_t test_vqrdmlshs_lane_s32(int32_t a, int32_t b, int32x2_t c) {
// CHECK: extractelement <2 x i32> {{%.*}}, i32 1
// CHECK: call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// CHECK: call i32 @llvm.aarch64.neon.sqsub.i32(i32 {{%.*}}, i32 {{%.*}})
  return vqrdmlshs_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: test_vqrdmlshh_laneq_s16
int16_t test_vqrdmlshh_laneq_s16(int16_t a, int16_t b, int16x8_t c) {
// CHECK: extractelement <8 x i16> {{%.*}}, i32 7
// CHECK: [[insb:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insc:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[mul:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[insb]], <4 x i16> [[insc]])
// CHECK: extractelement <4 x i16> [[mul]], i64 0
// CHECK: [[insa:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[insmul:%.*]] = insertelement <4 x i16> undef, i16 {{%.*}}, i64 0
// CHECK: [[sub:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> [[insa]], <4 x i16> [[insmul]])
// CHECK: extractelement <4 x i16> [[sub]], i64 0
  return vqrdmlshh_laneq_s16(a, b, c, 7);
}

// CHECK-LABEL: test_vqrdmlshs_laneq_s32
int32_t test_vqrdmlshs_laneq_s32(int32_t a, int32_t b, int32x4_t c) {
// CHECK: extractelement <4 x i32> {{%.*}}, i32 3
// CHECK: call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 {{%.*}}, i32 {{%.*}})
// CHECK: call i32 @llvm.aarch64.neon.sqsub.i32(i32 {{%.*}}, i32 {{%.*}})
  return vqrdmlshs_laneq_s32(a, b, c, 3);
}
