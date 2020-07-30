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

svbool_t test_svwhilege_b8_s32(int32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b8_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.whilege.nxv16i1.i32(i32 %op1, i32 %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b8'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b8_s32'}}
  return SVE_ACLE_FUNC(svwhilege_b8,_s32,,)(op1, op2);
}

svbool_t test_svwhilege_b16_s32(int32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b16_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilege.nxv8i1.i32(i32 %op1, i32 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b16'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b16_s32'}}
  return SVE_ACLE_FUNC(svwhilege_b16,_s32,,)(op1, op2);
}

svbool_t test_svwhilege_b32_s32(int32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b32_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilege.nxv4i1.i32(i32 %op1, i32 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b32'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b32_s32'}}
  return SVE_ACLE_FUNC(svwhilege_b32,_s32,,)(op1, op2);
}

svbool_t test_svwhilege_b64_s32(int32_t op1, int32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b64_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilege.nxv2i1.i32(i32 %op1, i32 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b64'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b64_s32'}}
  return SVE_ACLE_FUNC(svwhilege_b64,_s32,,)(op1, op2);
}

svbool_t test_svwhilege_b8_u32(uint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b8_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.whilehs.nxv16i1.i32(i32 %op1, i32 %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b8'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b8_u32'}}
  return SVE_ACLE_FUNC(svwhilege_b8,_u32,,)(op1, op2);
}

svbool_t test_svwhilege_b16_u32(uint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b16_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilehs.nxv8i1.i32(i32 %op1, i32 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b16'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b16_u32'}}
  return SVE_ACLE_FUNC(svwhilege_b16,_u32,,)(op1, op2);
}

svbool_t test_svwhilege_b32_u32(uint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b32_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilehs.nxv4i1.i32(i32 %op1, i32 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b32'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b32_u32'}}
  return SVE_ACLE_FUNC(svwhilege_b32,_u32,,)(op1, op2);
}

svbool_t test_svwhilege_b64_u32(uint32_t op1, uint32_t op2)
{
  // CHECK-LABEL: test_svwhilege_b64_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilehs.nxv2i1.i32(i32 %op1, i32 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b64'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b64_u32'}}
  return SVE_ACLE_FUNC(svwhilege_b64,_u32,,)(op1, op2);
}

svbool_t test_svwhilege_b8_s64(int64_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b8_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.whilege.nxv16i1.i64(i64 %op1, i64 %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b8'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b8_s64'}}
  return SVE_ACLE_FUNC(svwhilege_b8,_s64,,)(op1, op2);
}

svbool_t test_svwhilege_b16_s64(int64_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b16_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilege.nxv8i1.i64(i64 %op1, i64 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b16'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b16_s64'}}
  return SVE_ACLE_FUNC(svwhilege_b16,_s64,,)(op1, op2);
}

svbool_t test_svwhilege_b32_s64(int64_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b32_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilege.nxv4i1.i64(i64 %op1, i64 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b32'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b32_s64'}}
  return SVE_ACLE_FUNC(svwhilege_b32,_s64,,)(op1, op2);
}

svbool_t test_svwhilege_b64_s64(int64_t op1, int64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b64_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilege.nxv2i1.i64(i64 %op1, i64 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b64'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b64_s64'}}
  return SVE_ACLE_FUNC(svwhilege_b64,_s64,,)(op1, op2);
}

svbool_t test_svwhilege_b8_u64(uint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b8_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.whilehs.nxv16i1.i64(i64 %op1, i64 %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b8'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b8_u64'}}
  return SVE_ACLE_FUNC(svwhilege_b8,_u64,,)(op1, op2);
}

svbool_t test_svwhilege_b16_u64(uint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b16_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilehs.nxv8i1.i64(i64 %op1, i64 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b16'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b16_u64'}}
  return SVE_ACLE_FUNC(svwhilege_b16,_u64,,)(op1, op2);
}

svbool_t test_svwhilege_b32_u64(uint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b32_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilehs.nxv4i1.i64(i64 %op1, i64 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b32'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b32_u64'}}
  return SVE_ACLE_FUNC(svwhilege_b32,_u64,,)(op1, op2);
}

svbool_t test_svwhilege_b64_u64(uint64_t op1, uint64_t op2)
{
  // CHECK-LABEL: test_svwhilege_b64_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilehs.nxv2i1.i64(i64 %op1, i64 %op2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilege_b64'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilege_b64_u64'}}
  return SVE_ACLE_FUNC(svwhilege_b64,_u64,,)(op1, op2);
}
