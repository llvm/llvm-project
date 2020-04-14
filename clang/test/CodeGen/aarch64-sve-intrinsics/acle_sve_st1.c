// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svst1_s8(svbool_t pg, int8_t *base, svint8_t data)
{
  // CHECK-LABEL: test_svst1_s8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: call void @llvm.masked.store.nxv16i8.p0nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8>* %[[BASE]], i32 1, <vscale x 16 x i1> %pg)
  return SVE_ACLE_FUNC(svst1,_s8,,)(pg, base, data);
}

void test_svst1_s16(svbool_t pg, int16_t *base, svint16_t data)
{
  // CHECK-LABEL: test_svst1_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK: call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_s16,,)(pg, base, data);
}

void test_svst1_s32(svbool_t pg, int32_t *base, svint32_t data)
{
  // CHECK-LABEL: test_svst1_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK: call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_s32,,)(pg, base, data);
}

void test_svst1_s64(svbool_t pg, int64_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK: call void @llvm.masked.store.nxv2i64.p0nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_s64,,)(pg, base, data);
}

void test_svst1_u8(svbool_t pg, uint8_t *base, svuint8_t data)
{
  // CHECK-LABEL: test_svst1_u8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: call void @llvm.masked.store.nxv16i8.p0nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8>* %[[BASE]], i32 1, <vscale x 16 x i1> %pg)
  return SVE_ACLE_FUNC(svst1,_u8,,)(pg, base, data);
}

void test_svst1_u16(svbool_t pg, uint16_t *base, svuint16_t data)
{
  // CHECK-LABEL: test_svst1_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK: call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_u16,,)(pg, base, data);
}

void test_svst1_u32(svbool_t pg, uint32_t *base, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK: call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_u32,,)(pg, base, data);
}

void test_svst1_u64(svbool_t pg, uint64_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK: call void @llvm.masked.store.nxv2i64.p0nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_u64,,)(pg, base, data);
}

void test_svst1_f16(svbool_t pg, float16_t *base, svfloat16_t data)
{
  // CHECK-LABEL: test_svst1_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK: call void @llvm.masked.store.nxv8f16.p0nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x half>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_f16,,)(pg, base, data);
}

void test_svst1_f32(svbool_t pg, float32_t *base, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK: call void @llvm.masked.store.nxv4f32.p0nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x float>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_f32,,)(pg, base, data);
}

void test_svst1_f64(svbool_t pg, float64_t *base, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK: call void @llvm.masked.store.nxv2f64.p0nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x double>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1,_f64,,)(pg, base, data);
}

void test_svst1_vnum_s8(svbool_t pg, int8_t *base, int64_t vnum, svint8_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s8
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv16i8.p0nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8>* %[[GEP]], i32 1, <vscale x 16 x i1> %pg)
  return SVE_ACLE_FUNC(svst1_vnum,_s8,,)(pg, base, vnum, data);
}

void test_svst1_vnum_s16(svbool_t pg, int16_t *base, int64_t vnum, svint16_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_s16,,)(pg, base, vnum, data);
}

void test_svst1_vnum_s32(svbool_t pg, int32_t *base, int64_t vnum, svint32_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_s32,,)(pg, base, vnum, data);
}

void test_svst1_vnum_s64(svbool_t pg, int64_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i64.p0nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u8(svbool_t pg, uint8_t *base, int64_t vnum, svuint8_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u8
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv16i8.p0nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8>* %[[GEP]], i32 1, <vscale x 16 x i1> %pg)
  return SVE_ACLE_FUNC(svst1_vnum,_u8,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u16(svbool_t pg, uint16_t *base, int64_t vnum, svuint16_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_u16,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u32(svbool_t pg, uint32_t *base, int64_t vnum, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i32>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_u32,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u64(svbool_t pg, uint64_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i64.p0nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i64>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_u64,,)(pg, base, vnum, data);
}

void test_svst1_vnum_f16(svbool_t pg, float16_t *base, int64_t vnum, svfloat16_t data)
{
  // CHECK-LABEL: test_svst1_vnum_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv8f16.p0nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x half>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_f16,,)(pg, base, vnum, data);
}

void test_svst1_vnum_f32(svbool_t pg, float32_t *base, int64_t vnum, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_vnum_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv4f32.p0nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x float>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_f32,,)(pg, base, vnum, data);
}

void test_svst1_vnum_f64(svbool_t pg, float64_t *base, int64_t vnum, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_vnum_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2f64.p0nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x double>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  return SVE_ACLE_FUNC(svst1_vnum,_f64,,)(pg, base, vnum, data);
}
