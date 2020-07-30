// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svcmla_s8(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcmla_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 0)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s8'}}
  return SVE_ACLE_FUNC(svcmla,_s8,,)(op1, op2, op3, 0);
}

svint8_t test_svcmla_s8_1(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcmla_s8_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 90)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s8'}}
  return SVE_ACLE_FUNC(svcmla,_s8,,)(op1, op2, op3, 90);
}

svint8_t test_svcmla_s8_2(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcmla_s8_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 180)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s8'}}
  return SVE_ACLE_FUNC(svcmla,_s8,,)(op1, op2, op3, 180);
}

svint8_t test_svcmla_s8_3(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // CHECK-LABEL: test_svcmla_s8_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 270)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s8'}}
  return SVE_ACLE_FUNC(svcmla,_s8,,)(op1, op2, op3, 270);
}

svint16_t test_svcmla_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcmla_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s16'}}
  return SVE_ACLE_FUNC(svcmla,_s16,,)(op1, op2, op3, 0);
}

svint16_t test_svcmla_s16_1(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcmla_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s16'}}
  return SVE_ACLE_FUNC(svcmla,_s16,,)(op1, op2, op3, 90);
}

svint16_t test_svcmla_s16_2(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcmla_s16_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 180)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s16'}}
  return SVE_ACLE_FUNC(svcmla,_s16,,)(op1, op2, op3, 180);
}

svint16_t test_svcmla_s16_3(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcmla_s16_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 270)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s16'}}
  return SVE_ACLE_FUNC(svcmla,_s16,,)(op1, op2, op3, 270);
}

svint32_t test_svcmla_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svcmla_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s32'}}
  return SVE_ACLE_FUNC(svcmla,_s32,,)(op1, op2, op3, 0);
}

svint32_t test_svcmla_s32_1(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svcmla_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s32'}}
  return SVE_ACLE_FUNC(svcmla,_s32,,)(op1, op2, op3, 90);
}

svint32_t test_svcmla_s32_2(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svcmla_s32_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 180)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s32'}}
  return SVE_ACLE_FUNC(svcmla,_s32,,)(op1, op2, op3, 180);
}

svint32_t test_svcmla_s32_3(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svcmla_s32_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s32'}}
  return SVE_ACLE_FUNC(svcmla,_s32,,)(op1, op2, op3, 270);
}

svint64_t test_svcmla_s64(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svcmla_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s64'}}
  return SVE_ACLE_FUNC(svcmla,_s64,,)(op1, op2, op3, 0);
}

svint64_t test_svcmla_s64_1(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svcmla_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 90)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s64'}}
  return SVE_ACLE_FUNC(svcmla,_s64,,)(op1, op2, op3, 90);
}

svint64_t test_svcmla_s64_2(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svcmla_s64_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 180)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s64'}}
  return SVE_ACLE_FUNC(svcmla,_s64,,)(op1, op2, op3, 180);
}

svint64_t test_svcmla_s64_3(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // CHECK-LABEL: test_svcmla_s64_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 270)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_s64'}}
  return SVE_ACLE_FUNC(svcmla,_s64,,)(op1, op2, op3, 270);
}

svuint8_t test_svcmla_u8(svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svcmla_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 0)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u8'}}
  return SVE_ACLE_FUNC(svcmla,_u8,,)(op1, op2, op3, 0);
}

svuint8_t test_svcmla_u8_1(svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svcmla_u8_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 90)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u8'}}
  return SVE_ACLE_FUNC(svcmla,_u8,,)(op1, op2, op3, 90);
}

svuint8_t test_svcmla_u8_2(svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svcmla_u8_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 180)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u8'}}
  return SVE_ACLE_FUNC(svcmla,_u8,,)(op1, op2, op3, 180);
}

svuint8_t test_svcmla_u8_3(svuint8_t op1, svuint8_t op2, svuint8_t op3)
{
  // CHECK-LABEL: test_svcmla_u8_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cmla.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, <vscale x 16 x i8> %op3, i32 270)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u8'}}
  return SVE_ACLE_FUNC(svcmla,_u8,,)(op1, op2, op3, 270);
}

svuint16_t test_svcmla_u16(svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svcmla_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u16'}}
  return SVE_ACLE_FUNC(svcmla,_u16,,)(op1, op2, op3, 0);
}

svuint16_t test_svcmla_u16_1(svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svcmla_u16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u16'}}
  return SVE_ACLE_FUNC(svcmla,_u16,,)(op1, op2, op3, 90);
}

svuint16_t test_svcmla_u16_2(svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svcmla_u16_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 180)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u16'}}
  return SVE_ACLE_FUNC(svcmla,_u16,,)(op1, op2, op3, 180);
}

svuint16_t test_svcmla_u16_3(svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svcmla_u16_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 270)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u16'}}
  return SVE_ACLE_FUNC(svcmla,_u16,,)(op1, op2, op3, 270);
}

svuint32_t test_svcmla_u32(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svcmla_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u32'}}
  return SVE_ACLE_FUNC(svcmla,_u32,,)(op1, op2, op3, 0);
}

svuint32_t test_svcmla_u32_1(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svcmla_u32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u32'}}
  return SVE_ACLE_FUNC(svcmla,_u32,,)(op1, op2, op3, 90);
}

svuint32_t test_svcmla_u32_2(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svcmla_u32_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 180)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u32'}}
  return SVE_ACLE_FUNC(svcmla,_u32,,)(op1, op2, op3, 180);
}

svuint32_t test_svcmla_u32_3(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svcmla_u32_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u32'}}
  return SVE_ACLE_FUNC(svcmla,_u32,,)(op1, op2, op3, 270);
}

svuint64_t test_svcmla_u64(svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svcmla_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u64'}}
  return SVE_ACLE_FUNC(svcmla,_u64,,)(op1, op2, op3, 0);
}

svuint64_t test_svcmla_u64_1(svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svcmla_u64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 90)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u64'}}
  return SVE_ACLE_FUNC(svcmla,_u64,,)(op1, op2, op3, 90);
}

svuint64_t test_svcmla_u64_2(svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svcmla_u64_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 180)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u64'}}
  return SVE_ACLE_FUNC(svcmla,_u64,,)(op1, op2, op3, 180);
}

svuint64_t test_svcmla_u64_3(svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svcmla_u64_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cmla.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3, i32 270)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_u64'}}
  return SVE_ACLE_FUNC(svcmla,_u64,,)(op1, op2, op3, 270);
}

svint16_t test_svcmla_lane_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.lane.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_s16'}}
  return SVE_ACLE_FUNC(svcmla_lane,_s16,,)(op1, op2, op3, 0, 90);
}

svint16_t test_svcmla_lane_s16_1(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.lane.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 3, i32 180)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_s16'}}
  return SVE_ACLE_FUNC(svcmla_lane,_s16,,)(op1, op2, op3, 3, 180);
}

svint32_t test_svcmla_lane_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.lane.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_s32'}}
  return SVE_ACLE_FUNC(svcmla_lane,_s32,,)(op1, op2, op3, 0, 270);
}

svint32_t test_svcmla_lane_s32_1(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.lane.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 1, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_s32'}}
  return SVE_ACLE_FUNC(svcmla_lane,_s32,,)(op1, op2, op3, 1, 0);
}

svuint16_t test_svcmla_lane_u16(svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.lane.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 0, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_u16'}}
  return SVE_ACLE_FUNC(svcmla_lane,_u16,,)(op1, op2, op3, 0, 90);
}

svuint16_t test_svcmla_lane_u16_1(svuint16_t op1, svuint16_t op2, svuint16_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_u16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cmla.lane.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, <vscale x 8 x i16> %op3, i32 3, i32 180)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_u16'}}
  return SVE_ACLE_FUNC(svcmla_lane,_u16,,)(op1, op2, op3, 3, 180);
}

svuint32_t test_svcmla_lane_u32(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.lane.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 0, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_u32'}}
  return SVE_ACLE_FUNC(svcmla_lane,_u32,,)(op1, op2, op3, 0, 270);
}

svuint32_t test_svcmla_lane_u32_1(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svcmla_lane_u32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cmla.lane.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3, i32 1, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // expected-warning@+1 {{implicit declaration of function 'svcmla_lane_u32'}}
  return SVE_ACLE_FUNC(svcmla_lane,_u32,,)(op1, op2, op3, 1, 0);
}
