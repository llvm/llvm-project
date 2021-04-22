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

svint16_t test_svadalp_s16_z(svbool_t pg, svint16_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svadalp_s16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s16_z'}}
  return SVE_ACLE_FUNC(svadalp,_s16,_z,)(pg, op1, op2);
}

svint32_t test_svadalp_s32_z(svbool_t pg, svint32_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svadalp_s32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s32_z'}}
  return SVE_ACLE_FUNC(svadalp,_s32,_z,)(pg, op1, op2);
}

svint64_t test_svadalp_s64_z(svbool_t pg, svint64_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svadalp_s64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s64_z'}}
  return SVE_ACLE_FUNC(svadalp,_s64,_z,)(pg, op1, op2);
}

svuint16_t test_svadalp_u16_z(svbool_t pg, svuint16_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svadalp_u16_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sel.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 8 x i16> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %[[SEL]], <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u16_z'}}
  return SVE_ACLE_FUNC(svadalp,_u16,_z,)(pg, op1, op2);
}

svuint32_t test_svadalp_u32_z(svbool_t pg, svuint32_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svadalp_u32_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sel.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 4 x i32> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %[[SEL]], <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u32_z'}}
  return SVE_ACLE_FUNC(svadalp,_u32,_z,)(pg, op1, op2);
}

svuint64_t test_svadalp_u64_z(svbool_t pg, svuint64_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svadalp_u64_z
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SEL:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sel.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 2 x i64> zeroinitializer)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %[[SEL]], <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_z'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u64_z'}}
  return SVE_ACLE_FUNC(svadalp,_u64,_z,)(pg, op1, op2);
}

svint16_t test_svadalp_s16_m(svbool_t pg, svint16_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svadalp_s16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s16_m'}}
  return SVE_ACLE_FUNC(svadalp,_s16,_m,)(pg, op1, op2);
}

svint32_t test_svadalp_s32_m(svbool_t pg, svint32_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svadalp_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s32_m'}}
  return SVE_ACLE_FUNC(svadalp,_s32,_m,)(pg, op1, op2);
}

svint64_t test_svadalp_s64_m(svbool_t pg, svint64_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svadalp_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s64_m'}}
  return SVE_ACLE_FUNC(svadalp,_s64,_m,)(pg, op1, op2);
}

svuint16_t test_svadalp_u16_m(svbool_t pg, svuint16_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svadalp_u16_m
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u16_m'}}
  return SVE_ACLE_FUNC(svadalp,_u16,_m,)(pg, op1, op2);
}

svuint32_t test_svadalp_u32_m(svbool_t pg, svuint32_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svadalp_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u32_m'}}
  return SVE_ACLE_FUNC(svadalp,_u32,_m,)(pg, op1, op2);
}

svuint64_t test_svadalp_u64_m(svbool_t pg, svuint64_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svadalp_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_m'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u64_m'}}
  return SVE_ACLE_FUNC(svadalp,_u64,_m,)(pg, op1, op2);
}

svint16_t test_svadalp_s16_x(svbool_t pg, svint16_t op1, svint8_t op2)
{
  // CHECK-LABEL: test_svadalp_s16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sadalp.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s16_x'}}
  return SVE_ACLE_FUNC(svadalp,_s16,_x,)(pg, op1, op2);
}

svint32_t test_svadalp_s32_x(svbool_t pg, svint32_t op1, svint16_t op2)
{
  // CHECK-LABEL: test_svadalp_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s32_x'}}
  return SVE_ACLE_FUNC(svadalp,_s32,_x,)(pg, op1, op2);
}

svint64_t test_svadalp_s64_x(svbool_t pg, svint64_t op1, svint32_t op2)
{
  // CHECK-LABEL: test_svadalp_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sadalp.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_s64_x'}}
  return SVE_ACLE_FUNC(svadalp,_s64,_x,)(pg, op1, op2);
}

svuint16_t test_svadalp_u16_x(svbool_t pg, svuint16_t op1, svuint8_t op2)
{
  // CHECK-LABEL: test_svadalp_u16_x
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uadalp.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %op1, <vscale x 16 x i8> %op2)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u16_x'}}
  return SVE_ACLE_FUNC(svadalp,_u16,_x,)(pg, op1, op2);
}

svuint32_t test_svadalp_u32_x(svbool_t pg, svuint32_t op1, svuint16_t op2)
{
  // CHECK-LABEL: test_svadalp_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op1, <vscale x 8 x i16> %op2)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u32_x'}}
  return SVE_ACLE_FUNC(svadalp,_u32,_x,)(pg, op1, op2);
}

svuint64_t test_svadalp_u64_x(svbool_t pg, svuint64_t op1, svuint32_t op2)
{
  // CHECK-LABEL: test_svadalp_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uadalp.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op1, <vscale x 4 x i32> %op2)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svadalp_x'}}
  // expected-warning@+1 {{implicit declaration of function 'svadalp_u64_x'}}
  return SVE_ACLE_FUNC(svadalp,_u64,_x,)(pg, op1, op2);
}
