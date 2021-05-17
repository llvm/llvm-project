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

svint8_t test_svcadd_s8(svint8_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svcadd_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cadd.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, i32 90)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s8'}}
  return SVE_ACLE_FUNC(svcadd,_s8,,)(op1, op2, 90);
}

svint8_t test_svcadd_s8_1(svint8_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svcadd_s8_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cadd.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, i32 270)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s8'}}
  return SVE_ACLE_FUNC(svcadd,_s8,,)(op1, op2, 270);
}

svint16_t test_svcadd_s16(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svcadd_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cadd.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s16'}}
  return SVE_ACLE_FUNC(svcadd,_s16,,)(op1, op2, 90);
}

svint16_t test_svcadd_s16_1(svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svcadd_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cadd.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 270)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s16'}}
  return SVE_ACLE_FUNC(svcadd,_s16,,)(op1, op2, 270);
}

svint32_t test_svcadd_s32(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svcadd_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cadd.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s32'}}
  return SVE_ACLE_FUNC(svcadd,_s32,,)(op1, op2, 90);
}

svint32_t test_svcadd_s32_1(svint32_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svcadd_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cadd.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s32'}}
  return SVE_ACLE_FUNC(svcadd,_s32,,)(op1, op2, 270);
}

svint64_t test_svcadd_s64(svint64_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svcadd_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cadd.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 90)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s64'}}
  return SVE_ACLE_FUNC(svcadd,_s64,,)(op1, op2, 90);
}

svint64_t test_svcadd_s64_1(svint64_t op1, svint64_t op2)
{
  // CHECK-LABEL: test_svcadd_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cadd.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 270)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_s64'}}
  return SVE_ACLE_FUNC(svcadd,_s64,,)(op1, op2, 270);
}

svuint8_t test_svcadd_u8(svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svcadd_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cadd.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, i32 90)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u8'}}
  return SVE_ACLE_FUNC(svcadd,_u8,,)(op1, op2, 90);
}

svuint8_t test_svcadd_u8_1(svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svcadd_u8_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.cadd.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2, i32 270)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u8'}}
  return SVE_ACLE_FUNC(svcadd,_u8,,)(op1, op2, 270);
}

svuint16_t test_svcadd_u16(svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svcadd_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cadd.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 90)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u16'}}
  return SVE_ACLE_FUNC(svcadd,_u16,,)(op1, op2, 90);
}

svuint16_t test_svcadd_u16_1(svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svcadd_u16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.cadd.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2, i32 270)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u16'}}
  return SVE_ACLE_FUNC(svcadd,_u16,,)(op1, op2, 270);
}

svuint32_t test_svcadd_u32(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svcadd_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cadd.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 90)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u32'}}
  return SVE_ACLE_FUNC(svcadd,_u32,,)(op1, op2, 90);
}

svuint32_t test_svcadd_u32_1(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svcadd_u32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.cadd.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, i32 270)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u32'}}
  return SVE_ACLE_FUNC(svcadd,_u32,,)(op1, op2, 270);
}

svuint64_t test_svcadd_u64(svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svcadd_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cadd.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 90)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u64'}}
  return SVE_ACLE_FUNC(svcadd,_u64,,)(op1, op2, 90);
}

svuint64_t test_svcadd_u64_1(svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svcadd_u64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.cadd.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, i32 270)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svcadd'}}
  // expected-warning@+1 {{implicit declaration of function 'svcadd_u64'}}
  return SVE_ACLE_FUNC(svcadd,_u64,,)(op1, op2, 270);
}
