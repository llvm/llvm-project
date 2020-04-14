// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svld1_s8(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svld1_s8
  // CHECK-DAG: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK-DAG: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* %[[BASE]], i32 1, <vscale x 16 x i1> %pg, <vscale x 16 x i8> zeroinitializer)
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_s8,,)(pg, base);
}

svint16_t test_svld1_s16(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svld1_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> zeroinitializer)
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_s16,,)(pg, base);
}

svint32_t test_svld1_s32(svbool_t pg, const int32_t *base)
{
  // CHECK-LABEL: test_svld1_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> zeroinitializer)
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_s32,,)(pg, base);
}

svint64_t test_svld1_s64(svbool_t pg, const int64_t *base)
{
  // CHECK-LABEL: test_svld1_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> zeroinitializer)
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_s64,,)(pg, base);
}

svuint8_t test_svld1_u8(svbool_t pg, const uint8_t *base)
{
  // CHECK-LABEL: test_svld1_u8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* %[[BASE]], i32 1, <vscale x 16 x i1> %pg, <vscale x 16 x i8> zeroinitializer)
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_u8,,)(pg, base);
}

svuint16_t test_svld1_u16(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld1_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> zeroinitializer)
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_u16,,)(pg, base);
}

svuint32_t test_svld1_u32(svbool_t pg, const uint32_t *base)
{
  // CHECK-LABEL: test_svld1_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> zeroinitializer)
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_u32,,)(pg, base);
}

svuint64_t test_svld1_u64(svbool_t pg, const uint64_t *base)
{
  // CHECK-LABEL: test_svld1_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> zeroinitializer)
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_u64,,)(pg, base);
}

svfloat16_t test_svld1_f16(svbool_t pg, const float16_t *base)
{
  // CHECK-LABEL: test_svld1_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x half> @llvm.masked.load.nxv8f16.p0nxv8f16(<vscale x 8 x half>* %[[BASE]], i32 1, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> zeroinitializer)
  // CHECK: ret <vscale x 8 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_f16,,)(pg, base);
}

svfloat32_t test_svld1_f32(svbool_t pg, const float32_t *base)
{
  // CHECK-LABEL: test_svld1_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* %[[BASE]], i32 1, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> zeroinitializer)
  // CHECK: ret <vscale x 4 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_f32,,)(pg, base);
}

svfloat64_t test_svld1_f64(svbool_t pg, const float64_t *base)
{
  // CHECK-LABEL: test_svld1_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0nxv2f64(<vscale x 2 x double>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> zeroinitializer)
  // CHECK: ret <vscale x 2 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1,_f64,,)(pg, base);
}
svint8_t test_svld1_vnum_s8(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_s8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* %[[GEP]], i32 1, <vscale x 16 x i1> %pg, <vscale x 16 x i8> zeroinitializer)
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_s8,,)(pg, base, vnum);
}

svint16_t test_svld1_vnum_s16(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> zeroinitializer)
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_s16,,)(pg, base, vnum);
}

svint32_t test_svld1_vnum_s32(svbool_t pg, const int32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> zeroinitializer)
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_s32,,)(pg, base, vnum);
}

svint64_t test_svld1_vnum_s64(svbool_t pg, const int64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> zeroinitializer)
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_s64,,)(pg, base, vnum);
}

svuint8_t test_svld1_vnum_u8(svbool_t pg, const uint8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_u8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* %[[GEP]], i32 1, <vscale x 16 x i1> %pg, <vscale x 16 x i8> zeroinitializer)
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_u8,,)(pg, base, vnum);
}

svuint16_t test_svld1_vnum_u16(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> zeroinitializer)
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_u16,,)(pg, base, vnum);
}

svuint32_t test_svld1_vnum_u32(svbool_t pg, const uint32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> zeroinitializer)
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_u32,,)(pg, base, vnum);
}

svuint64_t test_svld1_vnum_u64(svbool_t pg, const uint64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> zeroinitializer)
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_u64,,)(pg, base, vnum);
}

svfloat16_t test_svld1_vnum_f16(svbool_t pg, const float16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x half> @llvm.masked.load.nxv8f16.p0nxv8f16(<vscale x 8 x half>* %[[GEP]], i32 1, <vscale x 8 x i1> %[[PG]], <vscale x 8 x half> zeroinitializer)
  // CHECK: ret <vscale x 8 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_f16,,)(pg, base, vnum);
}

svfloat32_t test_svld1_vnum_f32(svbool_t pg, const float32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* %[[GEP]], i32 1, <vscale x 4 x i1> %[[PG]], <vscale x 4 x float> zeroinitializer)
  // CHECK: ret <vscale x 4 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_f32,,)(pg, base, vnum);
}

svfloat64_t test_svld1_vnum_f64(svbool_t pg, const float64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1_vnum_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0nxv2f64(<vscale x 2 x double>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x double> zeroinitializer)
  // CHECK: ret <vscale x 2 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svld1_vnum,_f64,,)(pg, base, vnum);
}
