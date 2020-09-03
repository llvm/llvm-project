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

svbool_t test_svnmatch_s8(svbool_t pg, svint8_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svnmatch_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.nmatch.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svnmatch'}}
  // expected-warning@+1 {{implicit declaration of function 'svnmatch_s8'}}
  return SVE_ACLE_FUNC(svnmatch,_s8,,)(pg, op1, op2);
}

svbool_t test_svnmatch_s16(svbool_t pg, svint16_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svnmatch_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.nmatch.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: %[[RET:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[RET]]
  // overload-warning@+2 {{implicit declaration of function 'svnmatch'}}
  // expected-warning@+1 {{implicit declaration of function 'svnmatch_s16'}}
  return SVE_ACLE_FUNC(svnmatch,_s16,,)(pg, op1, op2);
}

svbool_t test_svnmatch_u8(svbool_t pg, svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svnmatch_u8
  // CHECK: %[[intrinsic:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.nmatch.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i1> %[[intrinsic]]
  // overload-warning@+2 {{implicit declaration of function 'svnmatch'}}
  // expected-warning@+1 {{implicit declaration of function 'svnmatch_u8'}}
  return SVE_ACLE_FUNC(svnmatch,_u8,,)(pg, op1, op2);
}

svbool_t test_svnmatch_u16(svbool_t pg, svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svnmatch_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.nmatch.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: %[[RET:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[RET]]
  // overload-warning@+2 {{implicit declaration of function 'svnmatch'}}
  // expected-warning@+1 {{implicit declaration of function 'svnmatch_u16'}}
  return SVE_ACLE_FUNC(svnmatch,_u16,,)(pg, op1, op2);
}
