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

svint16_t test_svqdmullb_s16(svint8_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svqdmullb_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullb.nxv8i16(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_s16'}}
  return SVE_ACLE_FUNC(svqdmullb,_s16,,)(op1, op2);
}

svint32_t test_svqdmullb_s32(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svqdmullb_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.nxv4i32(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_s32'}}
  return SVE_ACLE_FUNC(svqdmullb,_s32,,)(op1, op2);
}

svint64_t test_svqdmullb_s64(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svqdmullb_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.nxv2i64(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_s64'}}
  return SVE_ACLE_FUNC(svqdmullb,_s64,,)(op1, op2);
}

svint16_t test_svqdmullb_n_s16(svint8_t op1, int8_t op2)
{
  // CHECK-LABEL: test_svqdmullb_n_s16
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqdmullb.nxv8i16(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_n_s16'}}
  return SVE_ACLE_FUNC(svqdmullb,_n_s16,,)(op1, op2);
}

svint32_t test_svqdmullb_n_s32(svint16_t op1, int16_t op2)
{
  // CHECK-LABEL: test_svqdmullb_n_s32
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.nxv4i32(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_n_s32'}}
  return SVE_ACLE_FUNC(svqdmullb,_n_s32,,)(op1, op2);
}

svint64_t test_svqdmullb_n_s64(svint32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svqdmullb_n_s64
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.nxv2i64(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_n_s64'}}
  return SVE_ACLE_FUNC(svqdmullb,_n_s64,,)(op1, op2);
}

svint32_t test_svqdmullb_lane_s32(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svqdmullb_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.lane.nxv4i32(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_lane_s32'}}
  return SVE_ACLE_FUNC(svqdmullb_lane,_s32,,)(op1, op2, 0);
}

svint32_t test_svqdmullb_lane_s32_1(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svqdmullb_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmullb.lane.nxv4i32(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 7)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_lane_s32'}}
  return SVE_ACLE_FUNC(svqdmullb_lane,_s32,,)(op1, op2, 7);
}

svint64_t test_svqdmullb_lane_s64(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svqdmullb_lane_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.lane.nxv2i64(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_lane_s64'}}
  return SVE_ACLE_FUNC(svqdmullb_lane,_s64,,)(op1, op2, 0);
}

svint64_t test_svqdmullb_lane_s64_1(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svqdmullb_lane_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sqdmullb.lane.nxv2i64(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svqdmullb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svqdmullb_lane_s64'}}
  return SVE_ACLE_FUNC(svqdmullb_lane,_s64,,)(op1, op2, 3);
}
