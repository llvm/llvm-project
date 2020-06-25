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

svuint32_t test_svsbclt_u32(svuint32_t op1, svuint32_t op2, svuint32_t op3)
{
  // CHECK-LABEL: test_svsbclt_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sbclt.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %op3)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svsbclt'}}
  // expected-warning@+1 {{implicit declaration of function 'svsbclt_u32'}}
  return SVE_ACLE_FUNC(svsbclt,_u32,,)(op1, op2, op3);
}

svuint64_t test_svsbclt_u64(svuint64_t op1, svuint64_t op2, svuint64_t op3)
{
  // CHECK-LABEL: test_svsbclt_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sbclt.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %op3)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svsbclt'}}
  // expected-warning@+1 {{implicit declaration of function 'svsbclt_u64'}}
  return SVE_ACLE_FUNC(svsbclt,_u64,,)(op1, op2, op3);
}

svuint32_t test_svsbclt_n_u32(svuint32_t op1, svuint32_t op2, uint32_t op3)
{
  // CHECK-LABEL: test_svsbclt_n_u32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sbclt.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svsbclt'}}
  // expected-warning@+1 {{implicit declaration of function 'svsbclt_n_u32'}}
  return SVE_ACLE_FUNC(svsbclt,_n_u32,,)(op1, op2, op3);
}

svuint64_t test_svsbclt_n_u64(svuint64_t op1, svuint64_t op2, uint64_t op3)
{
  // CHECK-LABEL: test_svsbclt_n_u64
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op3)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sbclt.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svsbclt'}}
  // expected-warning@+1 {{implicit declaration of function 'svsbclt_n_u64'}}
  return SVE_ACLE_FUNC(svsbclt,_n_u64,,)(op1, op2, op3);
}
