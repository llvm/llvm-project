// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svst1h_s32(svbool_t pg, int16_t *base, svint32_t data)
{
  // CHECK-LABEL: test_svst1h_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  // CHECK: call void @llvm.masked.store.nxv4i16.p0nxv4i16(<vscale x 4 x i16> %[[DATA]], <vscale x 4 x i16>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h,_s32,,)(pg, base, data);
}

void test_svst1h_s64(svbool_t pg, int16_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1h_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  // CHECK: call void @llvm.masked.store.nxv2i16.p0nxv2i16(<vscale x 2 x i16> %[[DATA]], <vscale x 2 x i16>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h,_s64,,)(pg, base, data);
}

void test_svst1h_u32(svbool_t pg, uint16_t *base, svuint32_t data)
{
  // CHECK-LABEL: test_svst1h_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  // CHECK: call void @llvm.masked.store.nxv4i16.p0nxv4i16(<vscale x 4 x i16> %[[DATA]], <vscale x 4 x i16>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h,_u32,,)(pg, base, data);
}

void test_svst1h_u64(svbool_t pg, uint16_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1h_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  // CHECK: call void @llvm.masked.store.nxv2i16.p0nxv2i16(<vscale x 2 x i16> %[[DATA]], <vscale x 2 x i16>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h,_u64,,)(pg, base, data);
}

void test_svst1h_vnum_s32(svbool_t pg, int16_t *base, int64_t vnum, svint32_t data)
{
  // CHECK-LABEL: test_svst1h_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4i16.p0nxv4i16(<vscale x 4 x i16> %[[DATA]], <vscale x 4 x i16>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h_vnum,_s32,,)(pg, base, vnum, data);
}

void test_svst1h_vnum_s64(svbool_t pg, int16_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1h_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i16.p0nxv2i16(<vscale x 2 x i16> %[[DATA]], <vscale x 2 x i16>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1h_vnum_u32(svbool_t pg, uint16_t *base, int64_t vnum, svuint32_t data)
{
  // CHECK-LABEL: test_svst1h_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4i16.p0nxv4i16(<vscale x 4 x i16> %[[DATA]], <vscale x 4 x i16>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h_vnum,_u32,,)(pg, base, vnum, data);
}

void test_svst1h_vnum_u64(svbool_t pg, uint16_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1h_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i16.p0nxv2i16(<vscale x 2 x i16> %[[DATA]], <vscale x 2 x i16>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1h_vnum,_u64,,)(pg, base, vnum, data);
}
