// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svprfh(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 0)
  return svprfh(pg, base, SV_PLDL1KEEP);
}

void test_svprfh_1(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_1
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 1)
  return svprfh(pg, base, SV_PLDL1STRM);
}

void test_svprfh_2(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_2
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 2)
  return svprfh(pg, base, SV_PLDL2KEEP);
}

void test_svprfh_3(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_3
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 3)
  return svprfh(pg, base, SV_PLDL2STRM);
}

void test_svprfh_4(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_4
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 4)
  return svprfh(pg, base, SV_PLDL3KEEP);
}

void test_svprfh_5(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_5
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 5)
  return svprfh(pg, base, SV_PLDL3STRM);
}

void test_svprfh_6(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_6
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 8)
  return svprfh(pg, base, SV_PSTL1KEEP);
}

void test_svprfh_7(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_7
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 9)
  return svprfh(pg, base, SV_PSTL1STRM);
}

void test_svprfh_8(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_8
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 10)
  return svprfh(pg, base, SV_PSTL2KEEP);
}

void test_svprfh_9(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_9
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 11)
  return svprfh(pg, base, SV_PSTL2STRM);
}

void test_svprfh_10(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_10
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 12)
  return svprfh(pg, base, SV_PSTL3KEEP);
}

void test_svprfh_11(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfh_11
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %base, i32 13)
  return svprfh(pg, base, SV_PSTL3STRM);
}

void test_svprfh_vnum(svbool_t pg, const void *base, int64_t vnum)
{
  // CHECK-LABEL: test_svprfh_vnum
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum
  // CHECK-DAG: %[[I8_BASE:.*]] = bitcast <vscale x 8 x i16>* %[[GEP]] to i8*
  // CHECK: @llvm.aarch64.sve.prf.nxv8i1(<vscale x 8 x i1> %[[PG]], i8* %[[I8_BASE]], i32 0)
  return svprfh_vnum(pg, base, vnum, SV_PLDL1KEEP);
}
