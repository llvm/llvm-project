// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null 2>%t
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

void test_svst1b_s16(svbool_t pg, int8_t *base, svint16_t data)
{
  // CHECK-LABEL: test_svst1b_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i1> %[[PG]], i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_s16,,)(pg, base, data);
}

void test_svst1b_s32(svbool_t pg, int8_t *base, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i1> %[[PG]], i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_s32,,)(pg, base, data);
}

void test_svst1b_s64(svbool_t pg, int8_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i1> %[[PG]], i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_s64,,)(pg, base, data);
}

void test_svst1b_u16(svbool_t pg, uint8_t *base, svuint16_t data)
{
  // CHECK-LABEL: test_svst1b_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i1> %[[PG]], i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_u16,,)(pg, base, data);
}

void test_svst1b_u32(svbool_t pg, uint8_t *base, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i1> %[[PG]], i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_u32,,)(pg, base, data);
}

void test_svst1b_u64(svbool_t pg, uint8_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i1> %[[PG]], i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_u64,,)(pg, base, data);
}

void test_svst1b_vnum_s16(svbool_t pg, int8_t *base, int64_t vnum, svint16_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_s16,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_s32(svbool_t pg, int8_t *base, int64_t vnum, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_s32,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_s64(svbool_t pg, int8_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_u16(svbool_t pg, uint8_t *base, int64_t vnum, svuint16_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_u16,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_u32(svbool_t pg, uint8_t *base, int64_t vnum, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_u32,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_u64(svbool_t pg, uint8_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_u64,,)(pg, base, vnum, data);
}

void test_svst1b_scatter_u32base_s32(svbool_t pg, svuint32_t bases, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u32base_s32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u32base,,_s32)(pg, bases, data);
}

void test_svst1b_scatter_u64base_s64(svbool_t pg, svuint64_t bases, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u64base_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u64base,,_s64)(pg, bases, data);
}

void test_svst1b_scatter_u32base_u32(svbool_t pg, svuint32_t bases, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u32base_u32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u32base,,_u32)(pg, bases, data);
}

void test_svst1b_scatter_u64base_u64(svbool_t pg, svuint64_t bases, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u64base_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u64base,,_u64)(pg, bases, data);
}

void test_svst1b_scatter_s32offset_s32(svbool_t pg, int8_t *base, svint32_t offsets, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_s32offset_s32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i8(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,s32,offset,_s32)(pg, base, offsets, data);
}

void test_svst1b_scatter_s64offset_s64(svbool_t pg, int8_t *base, svint64_t offsets, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_s64offset_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i8(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,s64,offset,_s64)(pg, base, offsets, data);
}

void test_svst1b_scatter_s32offset_u32(svbool_t pg, uint8_t *base, svint32_t offsets, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_s32offset_u32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i8(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,s32,offset,_u32)(pg, base, offsets, data);
}

void test_svst1b_scatter_s64offset_u64(svbool_t pg, uint8_t *base, svint64_t offsets, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_s64offset_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i8(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,s64,offset,_u64)(pg, base, offsets, data);
}

void test_svst1b_scatter_u32offset_s32(svbool_t pg, int8_t *base, svuint32_t offsets, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u32offset_s32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i8(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], i8* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,u32,offset,_s32)(pg, base, offsets, data);
}

void test_svst1b_scatter_u64offset_s64(svbool_t pg, int8_t *base, svuint64_t offsets, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u64offset_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i8(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,u64,offset,_s64)(pg, base, offsets, data);
}

void test_svst1b_scatter_u32offset_u32(svbool_t pg, uint8_t *base, svuint32_t offsets, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u32offset_u32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i8(<vscale x 4 x i8> %0, <vscale x 4 x i1> %1, i8* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,u32,offset,_u32)(pg, base, offsets, data);
}

void test_svst1b_scatter_u64offset_u64(svbool_t pg, uint8_t *base, svuint64_t offsets, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u64offset_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i8(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], i8* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter_,u64,offset,_u64)(pg, base, offsets, data);
}

void test_svst1b_scatter_u32base_offset_s32(svbool_t pg, svuint32_t bases, int64_t offset, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u32base_offset_s32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u32base,_offset,_s32)(pg, bases, offset, data);
}

void test_svst1b_scatter_u64base_offset_s64(svbool_t pg, svuint64_t bases, int64_t offset, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u64base_offset_s64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u64base,_offset,_s64)(pg, bases, offset, data);
}

void test_svst1b_scatter_u32base_offset_u32(svbool_t pg, svuint32_t bases, int64_t offset, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u32base_offset_u32
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i8> %[[TRUNC]], <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u32base,_offset,_u32)(pg, bases, offset, data);
}

void test_svst1b_scatter_u64base_offset_u64(svbool_t pg, svuint64_t bases, int64_t offset, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_scatter_u64base_offset_u64
  // CHECK-DAG: %[[TRUNC:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i8> %[[TRUNC]], <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_scatter,_u64base,_offset,_u64)(pg, bases, offset, data);
}
