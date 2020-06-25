// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint32_t test_svexth_s32_z(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svexth_s32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sxth.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_s32,_z,)(pg, op);
}

svint64_t test_svexth_s64_z(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svexth_s64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sxth.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_s64,_z,)(pg, op);
}

svuint32_t test_svexth_u32_z(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svexth_u32_z
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uxth.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_u32,_z,)(pg, op);
}

svuint64_t test_svexth_u64_z(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svexth_u64_z
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uxth.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_u64,_z,)(pg, op);
}

svint32_t test_svexth_s32_m(svint32_t inactive, svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svexth_s32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sxth.nxv4i32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_s32,_m,)(inactive, pg, op);
}

svint64_t test_svexth_s64_m(svint64_t inactive, svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svexth_s64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sxth.nxv2i64(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_s64,_m,)(inactive, pg, op);
}

svuint32_t test_svexth_u32_m(svuint32_t inactive, svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svexth_u32_m
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uxth.nxv4i32(<vscale x 4 x i32> %inactive, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_u32,_m,)(inactive, pg, op);
}

svuint64_t test_svexth_u64_m(svuint64_t inactive, svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svexth_u64_m
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uxth.nxv2i64(<vscale x 2 x i64> %inactive, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_u64,_m,)(inactive, pg, op);
}

svint32_t test_svexth_s32_x(svbool_t pg, svint32_t op)
{
  // CHECK-LABEL: test_svexth_s32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sxth.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_s32,_x,)(pg, op);
}

svint64_t test_svexth_s64_x(svbool_t pg, svint64_t op)
{
  // CHECK-LABEL: test_svexth_s64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sxth.nxv2i64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_s64,_x,)(pg, op);
}

svuint32_t test_svexth_u32_x(svbool_t pg, svuint32_t op)
{
  // CHECK-LABEL: test_svexth_u32_x
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uxth.nxv4i32(<vscale x 4 x i32> undef, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %op)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_u32,_x,)(pg, op);
}

svuint64_t test_svexth_u64_x(svbool_t pg, svuint64_t op)
{
  // CHECK-LABEL: test_svexth_u64_x
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.uxth.nxv2i64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %op)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svexth,_u64,_x,)(pg, op);
}
