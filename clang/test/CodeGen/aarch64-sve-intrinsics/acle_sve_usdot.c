// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_MATMUL_INT8 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_MATMUL_INT8 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svint32_t test_svusdot_s32(svint32_t x, svuint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svusdot_s32
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusdot, _s32, , )(x, y, z);
}

svint32_t test_svusdot_n_s32(svint32_t x, svuint8_t y, int8_t z) {
  // CHECK-LABEL: test_svusdot_n_s32
  // CHECK: %[[SPLAT:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %z)
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %[[SPLAT]])
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusdot, _n_s32, , )(x, y, z);
}

svint32_t test_svusdot_lane_s32_0(svint32_t x, svuint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svusdot_lane_s32_0
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusdot_lane, _s32, , )(x, y, z, 0);
}

svint32_t test_svusdot_lane_s32_1(svint32_t x, svuint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svusdot_lane_s32_1
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusdot_lane, _s32, , )(x, y, z, 1);
}

svint32_t test_svusdot_lane_s32_2(svint32_t x, svuint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svusdot_lane_s32_2
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z, i32 2)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusdot_lane, _s32, , )(x, y, z, 2);
}

svint32_t test_svusdot_lane_s32_3(svint32_t x, svuint8_t y, svint8_t z) {
  // CHECK-LABEL: test_svusdot_lane_s32_3
  // CHECK: %[[RET:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.lane.nxv4i32(<vscale x 4 x i32> %x, <vscale x 16 x i8> %y, <vscale x 16 x i8> %z, i32 3)
  // CHECK: ret <vscale x 4 x i32> %[[RET]]
  return SVE_ACLE_FUNC(svusdot_lane, _s32, , )(x, y, z, 3);
}
