// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2-bitperm -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2-bitperm -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2-bitperm -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2-bitperm -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svuint8_t test_svbext_u8(svuint8_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svbext_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.bext.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_u8'}}
  return SVE_ACLE_FUNC(svbext,_u8,,)(op1, op2);
}

svuint16_t test_svbext_u16(svuint16_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svbext_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.bext.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_u16'}}
  return SVE_ACLE_FUNC(svbext,_u16,,)(op1, op2);
}

svuint32_t test_svbext_u32(svuint32_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svbext_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.bext.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_u32'}}
  return SVE_ACLE_FUNC(svbext,_u32,,)(op1, op2);
}

svuint64_t test_svbext_u64(svuint64_t op1, svuint64_t op2)
{
  // CHECK-LABEL: test_svbext_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.bext.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_u64'}}
  return SVE_ACLE_FUNC(svbext,_u64,,)(op1, op2);
}

svuint8_t test_svbext_n_u8(svuint8_t op1, uint8_t op2)
{
  // CHECK-LABEL: test_svbext_n_u8
  // CHECK: %[[DUP:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.bext.x.nxv16i8(<vscale x 16 x i8> %op1, <vscale x 16 x i8> %[[DUP]])
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_n_u8'}}
  return SVE_ACLE_FUNC(svbext,_n_u8,,)(op1, op2);
}

svuint16_t test_svbext_n_u16(svuint16_t op1, uint16_t op2)
{
  // CHECK-LABEL: test_svbext_n_u16
  // CHECK: %[[DUP:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.bext.x.nxv8i16(<vscale x 8 x i16> %op1, <vscale x 8 x i16> %[[DUP]])
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_n_u16'}}
  return SVE_ACLE_FUNC(svbext,_n_u16,,)(op1, op2);
}

svuint32_t test_svbext_n_u32(svuint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svbext_n_u32
  // CHECK: %[[DUP:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.bext.x.nxv4i32(<vscale x 4 x i32> %op1, <vscale x 4 x i32> %[[DUP]])
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_n_u32'}}
  return SVE_ACLE_FUNC(svbext,_n_u32,,)(op1, op2);
}

svuint64_t test_svbext_n_u64(svuint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svbext_n_u64
  // CHECK: %[[DUP:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %op2)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.bext.x.nxv2i64(<vscale x 2 x i64> %op1, <vscale x 2 x i64> %[[DUP]])
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svbext'}}
  // expected-warning@+1 {{implicit declaration of function 'svbext_n_u64'}}
  return SVE_ACLE_FUNC(svbext,_n_u64,,)(op1, op2);
}
