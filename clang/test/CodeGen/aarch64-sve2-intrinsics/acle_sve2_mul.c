// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint16_t test_svmul_lane_s16(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svmul_lane_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.lane.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_s16'}}
  return SVE_ACLE_FUNC(svmul_lane,_s16,,)(op1, op2, 0);
}

svint16_t test_svmul_lane_s16_1(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svmul_lane_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.lane.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 7)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_s16'}}
  return SVE_ACLE_FUNC(svmul_lane,_s16,,)(op1, op2, 7);
}

svint32_t test_svmul_lane_s32(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svmul_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_s32'}}
  return SVE_ACLE_FUNC(svmul_lane,_s32,,)(op1, op2, 0);
}

svint32_t test_svmul_lane_s32_1(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svmul_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_s32'}}
  return SVE_ACLE_FUNC(svmul_lane,_s32,,)(op1, op2, 3);
}

svint64_t test_svmul_lane_s64(svint64_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svmul_lane_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_s64'}}
  return SVE_ACLE_FUNC(svmul_lane,_s64,,)(op1, op2, 0);
}

svint64_t test_svmul_lane_s64_1(svint64_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svmul_lane_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_s64'}}
  return SVE_ACLE_FUNC(svmul_lane,_s64,,)(op1, op2, 1);
}

svuint16_t test_svmul_lane_u16(svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svmul_lane_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.lane.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_u16'}}
  return SVE_ACLE_FUNC(svmul_lane,_u16,,)(op1, op2, 0);
}

svuint16_t test_svmul_lane_u16_1(svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svmul_lane_u16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.lane.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 7)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_u16'}}
  return SVE_ACLE_FUNC(svmul_lane,_u16,,)(op1, op2, 7);
}

svuint32_t test_svmul_lane_u32(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svmul_lane_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_u32'}}
  return SVE_ACLE_FUNC(svmul_lane,_u32,,)(op1, op2, 0);
}

svuint32_t test_svmul_lane_u32_1(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svmul_lane_u32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_u32'}}
  return SVE_ACLE_FUNC(svmul_lane,_u32,,)(op1, op2, 3);
}

svuint64_t test_svmul_lane_u64(svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svmul_lane_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_u64'}}
  return SVE_ACLE_FUNC(svmul_lane,_u64,,)(op1, op2, 0);
}

svuint64_t test_svmul_lane_u64_1(svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svmul_lane_u64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 1)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svmul_lane_u64'}}
  return SVE_ACLE_FUNC(svmul_lane,_u64,,)(op1, op2, 1);
}
