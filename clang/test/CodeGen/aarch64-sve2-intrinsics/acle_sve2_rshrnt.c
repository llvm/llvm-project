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

svint8_t test_svrshrnt_n_s16(svint8_t op, svint16_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.rshrnt.nxv8i16(<vscale x 16 x i8> %op, <vscale x 8 x i16> %op1, i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_s16'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_s16,,)(op, op1, 1);
}

svint8_t test_svrshrnt_n_s16_1(svint8_t op, svint16_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_s16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.rshrnt.nxv8i16(<vscale x 16 x i8> %op, <vscale x 8 x i16> %op1, i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_s16'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_s16,,)(op, op1, 8);
}

svint16_t test_svrshrnt_n_s32(svint16_t op, svint32_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rshrnt.nxv4i32(<vscale x 8 x i16> %op, <vscale x 4 x i32> %op1, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_s32'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_s32,,)(op, op1, 1);
}

svint16_t test_svrshrnt_n_s32_1(svint16_t op, svint32_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_s32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rshrnt.nxv4i32(<vscale x 8 x i16> %op, <vscale x 4 x i32> %op1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_s32'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_s32,,)(op, op1, 16);
}

svint32_t test_svrshrnt_n_s64(svint32_t op, svint64_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.rshrnt.nxv2i64(<vscale x 4 x i32> %op, <vscale x 2 x i64> %op1, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_s64'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_s64,,)(op, op1, 1);
}

svint32_t test_svrshrnt_n_s64_1(svint32_t op, svint64_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_s64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.rshrnt.nxv2i64(<vscale x 4 x i32> %op, <vscale x 2 x i64> %op1, i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_s64'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_s64,,)(op, op1, 32);
}

svuint8_t test_svrshrnt_n_u16(svuint8_t op, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.rshrnt.nxv8i16(<vscale x 16 x i8> %op, <vscale x 8 x i16> %op1, i32 1)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_u16'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_u16,,)(op, op1, 1);
}

svuint8_t test_svrshrnt_n_u16_1(svuint8_t op, svuint16_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_u16_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.rshrnt.nxv8i16(<vscale x 16 x i8> %op, <vscale x 8 x i16> %op1, i32 8)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_u16'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_u16,,)(op, op1, 8);
}

svuint16_t test_svrshrnt_n_u32(svuint16_t op, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rshrnt.nxv4i32(<vscale x 8 x i16> %op, <vscale x 4 x i32> %op1, i32 1)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_u32'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_u32,,)(op, op1, 1);
}

svuint16_t test_svrshrnt_n_u32_1(svuint16_t op, svuint32_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_u32_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rshrnt.nxv4i32(<vscale x 8 x i16> %op, <vscale x 4 x i32> %op1, i32 16)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_u32'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_u32,,)(op, op1, 16);
}

svuint32_t test_svrshrnt_n_u64(svuint32_t op, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.rshrnt.nxv2i64(<vscale x 4 x i32> %op, <vscale x 2 x i64> %op1, i32 1)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_u64'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_u64,,)(op, op1, 1);
}

svuint32_t test_svrshrnt_n_u64_1(svuint32_t op, svuint64_t op1)
{
  // CHECK-LABEL: test_svrshrnt_n_u64_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.rshrnt.nxv2i64(<vscale x 4 x i32> %op, <vscale x 2 x i64> %op1, i32 32)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svrshrnt'}}
  // expected-warning@+1 {{implicit declaration of function 'svrshrnt_n_u64'}}
  return SVE_ACLE_FUNC(svrshrnt,_n_u64,,)(op, op1, 32);
}
