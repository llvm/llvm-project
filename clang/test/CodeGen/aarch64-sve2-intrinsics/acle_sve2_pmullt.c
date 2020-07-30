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

svuint8_t test_svpmullt_pair_u8(svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svpmullt_pair_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.pmullt.pair.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt_pair'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_pair_u8'}}
  return SVE_ACLE_FUNC(svpmullt_pair,_u8,,)(op1, op2);
}

svuint32_t test_svpmullt_pair_u32(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svpmullt_pair_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.pmullt.pair.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt_pair'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_pair_u32'}}
  return SVE_ACLE_FUNC(svpmullt_pair,_u32,,)(op1, op2);
}

svuint8_t test_svpmullt_pair_n_u8(svuint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svpmullt_pair_n_u8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.pmullt.pair.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt_pair'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_pair_n_u8'}}
  return SVE_ACLE_FUNC(svpmullt_pair,_n_u8,,)(op1, op2);
}

svuint32_t test_svpmullt_pair_n_u32(svuint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svpmullt_pair_n_u32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.pmullt.pair.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt_pair'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_pair_n_u32'}}
  return SVE_ACLE_FUNC(svpmullt_pair,_n_u32,,)(op1, op2);
}

svuint16_t test_svpmullt_u16(svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svpmullt_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.pmullt.pair.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: %[[BITCAST:.*]] = bitcast <vscale x 16 x i8> %[[INTRINSIC]] to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[BITCAST]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_u16'}}
  return SVE_ACLE_FUNC(svpmullt,_u16,,)(op1, op2);
}

svuint64_t test_svpmullt_u64(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svpmullt_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.pmullt.pair.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: %[[BITCAST:.*]] = bitcast <vscale x 4 x i32> %[[INTRINSIC]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[BITCAST]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_u64'}}
  return SVE_ACLE_FUNC(svpmullt,_u64,,)(op1, op2);
}

svuint16_t test_svpmullt_n_u16(svuint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svpmullt_n_u16
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.pmullt.pair.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: %[[BITCAST:.*]] = bitcast <vscale x 16 x i8> %[[INTRINSIC]] to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[BITCAST]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_n_u16'}}
  return SVE_ACLE_FUNC(svpmullt,_n_u16,,)(op1, op2);
}

svuint64_t test_svpmullt_n_u64(svuint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svpmullt_n_u64
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.pmullt.pair.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: %[[BITCAST:.*]] = bitcast <vscale x 4 x i32> %[[INTRINSIC]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[BITCAST]]
  // overload-warning@+2 {{implicit declaration of function 'svpmullt'}}
  // expected-warning@+1 {{implicit declaration of function 'svpmullt_n_u64'}}
  return SVE_ACLE_FUNC(svpmullt,_n_u64,,)(op1, op2);
}
