// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint32_t test_svcdot_s32(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcdot_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s32'}}
  return SVE_ACLE_FUNC(svcdot,_s32,,)(op1, op2, op3, 0);
}

svint32_t test_svcdot_s32_1(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcdot_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s32'}}
  return SVE_ACLE_FUNC(svcdot,_s32,,)(op1, op2, op3, 90);
}

svint32_t test_svcdot_s32_2(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcdot_s32_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 180)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s32'}}
  return SVE_ACLE_FUNC(svcdot,_s32,,)(op1, op2, op3, 180);
}

svint32_t test_svcdot_s32_3(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcdot_s32_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s32'}}
  return SVE_ACLE_FUNC(svcdot,_s32,,)(op1, op2, op3, 270);
}

svint64_t test_svcdot_s64(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcdot_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s64'}}
  return SVE_ACLE_FUNC(svcdot,_s64,,)(op1, op2, op3, 0);
}

svint64_t test_svcdot_s64_1(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcdot_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 90)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s64'}}
  return SVE_ACLE_FUNC(svcdot,_s64,,)(op1, op2, op3, 90);
}

svint64_t test_svcdot_s64_2(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcdot_s64_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 180)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s64'}}
  return SVE_ACLE_FUNC(svcdot,_s64,,)(op1, op2, op3, 180);
}

svint64_t test_svcdot_s64_3(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcdot_s64_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 270)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_s64'}}
  return SVE_ACLE_FUNC(svcdot,_s64,,)(op1, op2, op3, 270);
}

svint32_t test_svcdot_lane_s32(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcdot_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 0, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_lane_s32'}}
  return SVE_ACLE_FUNC(svcdot_lane,_s32,,)(op1, op2, op3, 0, 0);
}

svint32_t test_svcdot_lane_s32_1(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcdot_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cdot.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 2, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_lane_s32'}}
  return SVE_ACLE_FUNC(svcdot_lane,_s32,,)(op1, op2, op3, 2, 90);
}

svint64_t test_svcdot_lane_s64(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcdot_lane_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cdot.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0, i32 180)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcdot_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svcdot_lane_s64'}}
  return SVE_ACLE_FUNC(svcdot_lane,_s64,,)(op1, op2, op3, 0, 180);
}
