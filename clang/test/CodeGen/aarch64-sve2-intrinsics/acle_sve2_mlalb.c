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

svint16_t test_svmlalb_s16(svint16_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svmlalb_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.smlalb.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_s16'}}
  return SVE_ACLE_FUNC(svmlalb,_s16,,)(op1, op2, op3);
}

svint32_t test_svmlalb_s32(svint32_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_s32'}}
  return SVE_ACLE_FUNC(svmlalb,_s32,,)(op1, op2, op3);
}

svint64_t test_svmlalb_s64(svint64_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_s64'}}
  return SVE_ACLE_FUNC(svmlalb,_s64,,)(op1, op2, op3);
}

svuint16_t test_svmlalb_u16(svuint16_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svmlalb_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.umlalb.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_u16'}}
  return SVE_ACLE_FUNC(svmlalb,_u16,,)(op1, op2, op3);
}

svuint32_t test_svmlalb_u32(svuint32_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_u32'}}
  return SVE_ACLE_FUNC(svmlalb,_u32,,)(op1, op2, op3);
}

svuint64_t test_svmlalb_u64(svuint64_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_u64'}}
  return SVE_ACLE_FUNC(svmlalb,_u64,,)(op1, op2, op3);
}

svint16_t test_svmlalb_n_s16(svint16_t op1, svint8_t op2, int8_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_s16
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.smlalb.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_s16'}}
  return SVE_ACLE_FUNC(svmlalb,_n_s16,,)(op1, op2, op3);
}

svint32_t test_svmlalb_n_s32(svint32_t op1, svint16_t op2, int16_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_s32
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_s32'}}
  return SVE_ACLE_FUNC(svmlalb,_n_s32,,)(op1, op2, op3);
}

svint64_t test_svmlalb_n_s64(svint64_t op1, svint32_t op2, int32_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_s64
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_s64'}}
  return SVE_ACLE_FUNC(svmlalb,_n_s64,,)(op1, op2, op3);
}

svuint16_t test_svmlalb_n_u16(svuint16_t op1, svuint8_t op2, uint8_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_u16
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.umlalb.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_u16'}}
  return SVE_ACLE_FUNC(svmlalb,_n_u16,,)(op1, op2, op3);
}

svuint32_t test_svmlalb_n_u32(svuint32_t op1, svuint16_t op2, uint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_u32
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_u32'}}
  return SVE_ACLE_FUNC(svmlalb,_n_u32,,)(op1, op2, op3);
}

svuint64_t test_svmlalb_n_u64(svuint64_t op1, svuint32_t op2, uint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_u64
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_u64'}}
  return SVE_ACLE_FUNC(svmlalb,_n_u64,,)(op1, op2, op3);
}

svint32_t test_svmlalb_lane_s32(svint32_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_s32'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_s32,,)(op1, op2, op3, 0);
}

svint32_t test_svmlalb_lane_s32_1(svint32_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.smlalb.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 7)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_s32'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_s32,,)(op1, op2, op3, 7);
}

svint64_t test_svmlalb_lane_s64(svint64_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_s64'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_s64,,)(op1, op2, op3, 0);
}

svint64_t test_svmlalb_lane_s64_1(svint64_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.smlalb.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_s64'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_s64,,)(op1, op2, op3, 3);
}

svuint32_t test_svmlalb_lane_u32(svuint32_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_u32'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_u32,,)(op1, op2, op3, 0);
}

svuint32_t test_svmlalb_lane_u32_1(svuint32_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_u32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.umlalb.lane.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 7)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_u32'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_u32,,)(op1, op2, op3, 7);
}

svuint64_t test_svmlalb_lane_u64(svuint64_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_u64'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_u64,,)(op1, op2, op3, 0);
}

svuint64_t test_svmlalb_lane_u64_1(svuint64_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_u64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.umlalb.lane.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_u64'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_u64,,)(op1, op2, op3, 3);
}

svfloat32_t test_svmlalb_f32(svfloat32_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmlalb_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.nxv4f32(<vscale x 4 x float> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_f32'}}
  return SVE_ACLE_FUNC(svmlalb,_f32,,)(op1, op2, op3);
}

svfloat32_t test_svmlalb_n_f32(svfloat32_t op1, svfloat16_t op2, float16_t op3)
{
  // CHECK-LABEL: test_svmlalb_n_f32
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.nxv4f32(<vscale x 4 x float> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %[[DUP]])
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_n_f32'}}
  return SVE_ACLE_FUNC(svmlalb,_n_f32,,)(op1, op2, op3);
}

svfloat32_t test_svmlalb_lane_f32(svfloat32_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.lane.nxv4f32(<vscale x 4 x float> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 0)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_f32'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_f32,,)(op1, op2, op3, 0);
}

svfloat32_t test_svmlalb_lane_f32_1(svfloat32_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // CHECK-LABEL: test_svmlalb_lane_f32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmlalb.lane.nxv4f32(<vscale x 4 x float> %op1, <vscale x 8 x half> %op2, <vscale x 8 x half> %op3, i32 7)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmlalb_lane'}}
  // expected-warning@+1 {{implicit declaration of function 'svmlalb_lane_f32'}}
  return SVE_ACLE_FUNC(svmlalb_lane,_f32,,)(op1, op2, op3, 7);
}
