// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svqrdcmlah_s8(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdcmlah.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 0)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s8'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 0);
}

svint8_t test_svqrdcmlah_s8_1(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s8_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdcmlah.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 90)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s8'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 90);
}

svint8_t test_svqrdcmlah_s8_2(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s8_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdcmlah.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 180)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s8'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 180);
}

svint8_t test_svqrdcmlah_s8_3(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s8_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrdcmlah.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 270)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s8'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 270);
}

svint16_t test_svqrdcmlah_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s16'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s16,,)(op1, op2, op3, 0);
}

svint16_t test_svqrdcmlah_s16_1(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s16'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s16,,)(op1, op2, op3, 90);
}

svint16_t test_svqrdcmlah_s16_2(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s16_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 180)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s16'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s16,,)(op1, op2, op3, 180);
}

svint16_t test_svqrdcmlah_s16_3(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s16_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 270)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s16'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s16,,)(op1, op2, op3, 270);
}

svint32_t test_svqrdcmlah_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s32'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s32,,)(op1, op2, op3, 0);
}

svint32_t test_svqrdcmlah_s32_1(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s32'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s32,,)(op1, op2, op3, 90);
}

svint32_t test_svqrdcmlah_s32_2(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s32_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 180)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s32'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s32,,)(op1, op2, op3, 180);
}

svint32_t test_svqrdcmlah_s32_3(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s32_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s32'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s32,,)(op1, op2, op3, 270);
}

svint64_t test_svqrdcmlah_s64(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdcmlah.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s64'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s64,,)(op1, op2, op3, 0);
}

svint64_t test_svqrdcmlah_s64_1(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdcmlah.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 90)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s64'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s64,,)(op1, op2, op3, 90);
}

svint64_t test_svqrdcmlah_s64_2(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s64_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdcmlah.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 180)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s64'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s64,,)(op1, op2, op3, 180);
}

svint64_t test_svqrdcmlah_s64_3(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_s64_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqrdcmlah.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 270)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_s64'}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s64,,)(op1, op2, op3, 270);
}

svint16_t test_svqrdcmlah_lane_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_lane_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_lane_s16'}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, 0, 0);
}

svint16_t test_svqrdcmlah_lane_s16_1(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_lane_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 3, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_lane_s16'}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, 3, 90);
}

svint32_t test_svqrdcmlah_lane_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0, i32 180)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_lane_s32'}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s32,,)(op1, op2, op3, 0, 180);
}

svint32_t test_svqrdcmlah_lane_s32_1(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svqrdcmlah_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdcmlah.lane.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 1, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqrdcmlah_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqrdcmlah_lane_s32'}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s32,,)(op1, op2, op3, 1, 270);
}
