// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s | FileCheck %s

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
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK: call void @llvm.masked.store.nxv8i8.p0nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i8>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_s16,,)(pg, base, data);
}

void test_svst1b_s32(svbool_t pg, int8_t *base, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK: call void @llvm.masked.store.nxv4i8.p0nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i8>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_s32,,)(pg, base, data);
}

void test_svst1b_s64(svbool_t pg, int8_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK: call void @llvm.masked.store.nxv2i8.p0nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i8>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_s64,,)(pg, base, data);
}

void test_svst1b_u16(svbool_t pg, uint8_t *base, svuint16_t data)
{
  // CHECK-LABEL: test_svst1b_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK: call void @llvm.masked.store.nxv8i8.p0nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i8>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_u16,,)(pg, base, data);
}

void test_svst1b_u32(svbool_t pg, uint8_t *base, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK: call void @llvm.masked.store.nxv4i8.p0nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i8>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_u32,,)(pg, base, data);
}

void test_svst1b_u64(svbool_t pg, uint8_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK: call void @llvm.masked.store.nxv2i8.p0nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i8>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b,_u64,,)(pg, base, data);
}

void test_svst1b_vnum_s16(svbool_t pg, int8_t *base, int64_t vnum, svint16_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv8i8.p0nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i8>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_s16,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_s32(svbool_t pg, int8_t *base, int64_t vnum, svint32_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4i8.p0nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i8>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_s32,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_s64(svbool_t pg, int8_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i8.p0nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i8>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_u16(svbool_t pg, uint8_t *base, int64_t vnum, svuint16_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv8i8.p0nxv8i8(<vscale x 8 x i8> %[[DATA]], <vscale x 8 x i8>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_u16,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_u32(svbool_t pg, uint8_t *base, int64_t vnum, svuint32_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4i8.p0nxv4i8(<vscale x 4 x i8> %[[DATA]], <vscale x 4 x i8>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_u32,,)(pg, base, vnum, data);
}

void test_svst1b_vnum_u64(svbool_t pg, uint8_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1b_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i8.p0nxv2i8(<vscale x 2 x i8> %[[DATA]], <vscale x 2 x i8>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1b_vnum,_u64,,)(pg, base, vnum, data);
}
