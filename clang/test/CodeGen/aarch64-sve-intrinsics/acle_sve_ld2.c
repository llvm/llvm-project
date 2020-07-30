// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8x2_t test_svld2_s8(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svld2_s8
  // CHECK: %[[LOAD:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1(<vscale x 16 x i1> %pg, i8* %base)
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_s8,,)(pg, base);
}

svint16x2_t test_svld2_s16(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svld2_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1(<vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_s16,,)(pg, base);
}

svint32x2_t test_svld2_s32(svbool_t pg, const int32_t *base)
{
  // CHECK-LABEL: test_svld2_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1(<vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_s32,,)(pg, base);
}

svint64x2_t test_svld2_s64(svbool_t pg, const int64_t *base)
{
  // CHECK-LABEL: test_svld2_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1(<vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_s64,,)(pg, base);
}

svuint8x2_t test_svld2_u8(svbool_t pg, const uint8_t *base)
{
  // CHECK-LABEL: test_svld2_u8
  // CHECK: %[[LOAD:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1(<vscale x 16 x i1> %pg, i8* %base)
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_u8,,)(pg, base);
}

svuint16x2_t test_svld2_u16(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld2_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1(<vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_u16,,)(pg, base);
}

svuint32x2_t test_svld2_u32(svbool_t pg, const uint32_t *base)
{
  // CHECK-LABEL: test_svld2_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1(<vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_u32,,)(pg, base);
}

svuint64x2_t test_svld2_u64(svbool_t pg, const uint64_t *base)
{
  // CHECK-LABEL: test_svld2_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1(<vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_u64,,)(pg, base);
}

svfloat16x2_t test_svld2_f16(svbool_t pg, const float16_t *base)
{
  // CHECK-LABEL: test_svld2_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x half> @llvm.aarch64.sve.ld2.nxv16f16.nxv8i1(<vscale x 8 x i1> %[[PG]], half* %base)
  // CHECK-NEXT: ret <vscale x 16 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_f16,,)(pg, base);
}

svfloat32x2_t test_svld2_f32(svbool_t pg, const float32_t *base)
{
  // CHECK-LABEL: test_svld2_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x float> @llvm.aarch64.sve.ld2.nxv8f32.nxv4i1(<vscale x 4 x i1> %[[PG]], float* %base)
  // CHECK-NEXT: ret <vscale x 8 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_f32,,)(pg, base);
}

svfloat64x2_t test_svld2_f64(svbool_t pg, const float64_t *base)
{
  // CHECK-LABEL: test_svld2_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x double> @llvm.aarch64.sve.ld2.nxv4f64.nxv2i1(<vscale x 2 x i1> %[[PG]], double* %base)
  // CHECK-NEXT: ret <vscale x 4 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2,_f64,,)(pg, base);
}

svint8x2_t test_svld2_vnum_s8(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_s8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1(<vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_s8,,)(pg, base, vnum);
}

svint16x2_t test_svld2_vnum_s16(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1(<vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_s16,,)(pg, base, vnum);
}

svint32x2_t test_svld2_vnum_s32(svbool_t pg, const int32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1(<vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_s32,,)(pg, base, vnum);
}

svint64x2_t test_svld2_vnum_s64(svbool_t pg, const int64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1(<vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_s64,,)(pg, base, vnum);
}

svuint8x2_t test_svld2_vnum_u8(svbool_t pg, const uint8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_u8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1(<vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 32 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_u8,,)(pg, base, vnum);
}

svuint16x2_t test_svld2_vnum_u16(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1(<vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 16 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_u16,,)(pg, base, vnum);
}

svuint32x2_t test_svld2_vnum_u32(svbool_t pg, const uint32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1(<vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 8 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_u32,,)(pg, base, vnum);
}

svuint64x2_t test_svld2_vnum_u64(svbool_t pg, const uint64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1(<vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 4 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_u64,,)(pg, base, vnum);
}

svfloat16x2_t test_svld2_vnum_f16(svbool_t pg, const float16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x half> @llvm.aarch64.sve.ld2.nxv16f16.nxv8i1(<vscale x 8 x i1> %[[PG]], half* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 16 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_f16,,)(pg, base, vnum);
}

svfloat32x2_t test_svld2_vnum_f32(svbool_t pg, const float32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x float> @llvm.aarch64.sve.ld2.nxv8f32.nxv4i1(<vscale x 4 x i1> %[[PG]], float* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 8 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_f32,,)(pg, base, vnum);
}

svfloat64x2_t test_svld2_vnum_f64(svbool_t pg, const float64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld2_vnum_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x double> @llvm.aarch64.sve.ld2.nxv4f64.nxv2i1(<vscale x 2 x i1> %[[PG]], double* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 4 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svld2_vnum,_f64,,)(pg, base, vnum);
}
