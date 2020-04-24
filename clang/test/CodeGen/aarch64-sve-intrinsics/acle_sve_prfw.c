// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svprfw(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 0)
  return svprfw(pg, base, SV_PLDL1KEEP);
}

void test_svprfw_1(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_1
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 1)
  return svprfw(pg, base, SV_PLDL1STRM);
}

void test_svprfw_2(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_2
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 2)
  return svprfw(pg, base, SV_PLDL2KEEP);
}

void test_svprfw_3(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_3
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 3)
  return svprfw(pg, base, SV_PLDL2STRM);
}

void test_svprfw_4(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_4
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 4)
  return svprfw(pg, base, SV_PLDL3KEEP);
}

void test_svprfw_5(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_5
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 5)
  return svprfw(pg, base, SV_PLDL3STRM);
}

void test_svprfw_6(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_6
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 8)
  return svprfw(pg, base, SV_PSTL1KEEP);
}

void test_svprfw_7(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_7
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 9)
  return svprfw(pg, base, SV_PSTL1STRM);
}

void test_svprfw_8(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_8
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 10)
  return svprfw(pg, base, SV_PSTL2KEEP);
}

void test_svprfw_9(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_9
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 11)
  return svprfw(pg, base, SV_PSTL2STRM);
}

void test_svprfw_10(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_10
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 12)
  return svprfw(pg, base, SV_PSTL3KEEP);
}

void test_svprfw_11(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfw_11
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %base, i32 13)
  return svprfw(pg, base, SV_PSTL3STRM);
}

void test_svprfw_vnum(svbool_t pg, const void *base, int64_t vnum)
{
  // CHECK-LABEL: test_svprfw_vnum
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum
  // CHECK-DAG: %[[I8_BASE:.*]] = bitcast <vscale x 4 x i32>* %[[GEP]] to i8*
  // CHECK: @llvm.aarch64.sve.prf.nxv4i1(<vscale x 4 x i1> %[[PG]], i8* %[[I8_BASE]], i32 0)
  return svprfw_vnum(pg, base, vnum, SV_PLDL1KEEP);
}
