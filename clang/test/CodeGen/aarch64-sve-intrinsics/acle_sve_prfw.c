// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
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

void test_svprfw_gather_u32base(svbool_t pg, svuint32_t bases)
{
  // CHECK-LABEL: test_svprfw_gather_u32base
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather,_u32base,,)(pg, bases, SV_PLDL1KEEP);
}

void test_svprfw_gather_u64base(svbool_t pg, svuint64_t bases)
{
  // CHECK-LABEL: test_svprfw_gather_u64base
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather,_u64base,,)(pg, bases, SV_PLDL1KEEP);
}

void test_svprfw_gather_s32index(svbool_t pg, const void *base, svint32_t indices)
{
  // CHECK-LABEL: test_svprfw_gather_s32index
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.sxtw.index.nxv4i32(<vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %indices, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather_,s32,index,)(pg, base, indices, SV_PLDL1KEEP);
}

void test_svprfw_gather_s64index(svbool_t pg, const void *base, svint64_t indices)
{
  // CHECK-LABEL: test_svprfw_gather_s64index
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.index.nxv2i64(<vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %indices, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather_,s64,index,)(pg, base, indices, SV_PLDL1KEEP);
}

void test_svprfw_gather_u32index(svbool_t pg, const void *base, svuint32_t indices)
{
  // CHECK-LABEL: test_svprfw_gather_u32index
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.uxtw.index.nxv4i32(<vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %indices, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather_,u32,index,)(pg, base, indices, SV_PLDL1KEEP);
}

void test_svprfw_gather_u64index(svbool_t pg, const void *base, svuint64_t indices)
{
  // CHECK-LABEL: test_svprfw_gather_u64index
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.index.nxv2i64(<vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %indices, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather_,u64,index,)(pg, base, indices, SV_PLDL1KEEP);
}

void test_svprfw_gather_u32base_index(svbool_t pg, svuint32_t bases, int64_t index)
{
  // CHECK-LABEL: test_svprfw_gather_u32base_index
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %[[SHL]], i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather,_u32base,_index,)(pg, bases, index, SV_PLDL1KEEP);
}

void test_svprfw_gather_u64base_index(svbool_t pg, svuint64_t bases, int64_t index)
{
  // CHECK-LABEL: test_svprfw_gather_u64base_index
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.prfw.gather.scalar.offset.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]], i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfw_gather,_u64base,_index,)(pg, bases, index, SV_PLDL1KEEP);
}
