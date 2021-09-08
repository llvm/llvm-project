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

void test_svprfb(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 0)
  return svprfb(pg, base, SV_PLDL1KEEP);
}

void test_svprfb_1(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_1
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 1)
  return svprfb(pg, base, SV_PLDL1STRM);
}

void test_svprfb_2(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_2
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 2)
  return svprfb(pg, base, SV_PLDL2KEEP);
}

void test_svprfb_3(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_3
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 3)
  return svprfb(pg, base, SV_PLDL2STRM);
}

void test_svprfb_4(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_4
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 4)
  return svprfb(pg, base, SV_PLDL3KEEP);
}

void test_svprfb_5(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_5
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 5)
  return svprfb(pg, base, SV_PLDL3STRM);
}

void test_svprfb_6(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_6
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 8)
  return svprfb(pg, base, SV_PSTL1KEEP);
}

void test_svprfb_7(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_7
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 9)
  return svprfb(pg, base, SV_PSTL1STRM);
}

void test_svprfb_8(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_8
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 10)
  return svprfb(pg, base, SV_PSTL2KEEP);
}

void test_svprfb_9(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_9
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 11)
  return svprfb(pg, base, SV_PSTL2STRM);
}

void test_svprfb_10(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_10
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 12)
  return svprfb(pg, base, SV_PSTL3KEEP);
}

void test_svprfb_11(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_11
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 13)
  return svprfb(pg, base, SV_PSTL3STRM);
}

void test_svprfb_vnum(svbool_t pg, const void *base, int64_t vnum)
{
  // CHECK-LABEL: test_svprfb_vnum
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %[[GEP]], i32 0)
  return svprfb_vnum(pg, base, vnum, SV_PLDL1KEEP);
}

void test_svprfb_gather_u32base(svbool_t pg, svuint32_t bases)
{
  // CHECK-LABEL: test_svprfb_gather_u32base
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather,_u32base,,)(pg, bases, SV_PLDL1KEEP);
}

void test_svprfb_gather_u64base(svbool_t pg, svuint64_t bases)
{
  // CHECK-LABEL: test_svprfb_gather_u64base
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather,_u64base,,)(pg, bases, SV_PLDL1KEEP);
}

void test_svprfb_gather_s32offset(svbool_t pg, const void *base, svint32_t offsets)
{
  // CHECK-LABEL: test_svprfb_gather_s32offset
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.sxtw.index.nxv4i32(<vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %offsets, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather_,s32,offset,)(pg, base, offsets, SV_PLDL1KEEP);
}

void test_svprfb_gather_s64offset(svbool_t pg, const void *base, svint64_t offsets)
{
  // CHECK-LABEL: test_svprfb_gather_s64offset
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.index.nxv2i64(<vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %offsets, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather_,s64,offset,)(pg, base, offsets, SV_PLDL1KEEP);
}

void test_svprfb_gather_u32offset(svbool_t pg, const void *base, svuint32_t offsets)
{
  // CHECK-LABEL: test_svprfb_gather_u32offset
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.uxtw.index.nxv4i32(<vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %offsets, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather_,u32,offset,)(pg, base, offsets, SV_PLDL1KEEP);
}

void test_svprfb_gather_u64offset(svbool_t pg, const void *base, svuint64_t offsets)
{
  // CHECK-LABEL: test_svprfb_gather_u64offset
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.index.nxv2i64(<vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %offsets, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather_,u64,offset,)(pg, base, offsets, SV_PLDL1KEEP);
}

void test_svprfb_gather_u32base_offset(svbool_t pg, svuint32_t bases, int64_t offset)
{
  // CHECK-LABEL: test_svprfb_gather_u32base_offset
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset, i32 0)
  // CHECK: ret void
  return svprfb_gather_u32base_offset(pg, bases, offset, SV_PLDL1KEEP);
  return SVE_ACLE_FUNC(svprfb_gather,_u32base,_offset,)(pg, bases, offset, SV_PLDL1KEEP);
}

void test_svprfb_gather_u64base_offset(svbool_t pg, svuint64_t bases, int64_t offset)
{
  // CHECK-LABEL: test_svprfb_gather_u64base_offset
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.prfb.gather.scalar.offset.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset, i32 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svprfb_gather,_u64base,_offset,)(pg, bases, offset, SV_PLDL1KEEP);
}
