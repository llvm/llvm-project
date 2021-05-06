// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null
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
  // CHECK: call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_s8,,)(pg, base, data);
}

void test_svst1_s16(svbool_t pg, int16_t *base, svint16_t data)
{
  // CHECK-LABEL: test_svst1_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_s16,,)(pg, base, data);
}

void test_svst1_s32(svbool_t pg, int32_t *base, svint32_t data)
{
  // CHECK-LABEL: test_svst1_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_s32,,)(pg, base, data);
}

void test_svst1_s64(svbool_t pg, int64_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_s64,,)(pg, base, data);
}

void test_svst1_u8(svbool_t pg, uint8_t *base, svuint8_t data)
{
  // CHECK-LABEL: test_svst1_u8
  // CHECK: call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_u8,,)(pg, base, data);
}

void test_svst1_u16(svbool_t pg, uint16_t *base, svuint16_t data)
{
  // CHECK-LABEL: test_svst1_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_u16,,)(pg, base, data);
}

void test_svst1_u32(svbool_t pg, uint32_t *base, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_u32,,)(pg, base, data);
}

void test_svst1_u64(svbool_t pg, uint64_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_u64,,)(pg, base, data);
}

void test_svst1_f16(svbool_t pg, float16_t *base, svfloat16_t data)
{
  // CHECK-LABEL: test_svst1_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %[[PG]], half* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_f16,,)(pg, base, data);
}

void test_svst1_f32(svbool_t pg, float32_t *base, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_f32,,)(pg, base, data);
}

void test_svst1_f64(svbool_t pg, float64_t *base, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %base)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1,_f64,,)(pg, base, data);
}

void test_svst1_vnum_s8(svbool_t pg, int8_t *base, int64_t vnum, svint8_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_s8,,)(pg, base, vnum, data);
}

void test_svst1_vnum_s16(svbool_t pg, int16_t *base, int64_t vnum, svint16_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_s16,,)(pg, base, vnum, data);
}

void test_svst1_vnum_s32(svbool_t pg, int32_t *base, int64_t vnum, svint32_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_s32,,)(pg, base, vnum, data);
}

void test_svst1_vnum_s64(svbool_t pg, int64_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u8(svbool_t pg, uint8_t *base, int64_t vnum, svuint8_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u8
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_u8,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u16(svbool_t pg, uint16_t *base, int64_t vnum, svuint16_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_u16,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u32(svbool_t pg, uint32_t *base, int64_t vnum, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_u32,,)(pg, base, vnum, data);
}

void test_svst1_vnum_u64(svbool_t pg, uint64_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_u64,,)(pg, base, vnum, data);
}

void test_svst1_vnum_f16(svbool_t pg, float16_t *base, int64_t vnum, svfloat16_t data)
{
  // CHECK-LABEL: test_svst1_vnum_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %[[PG]], half* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_f16,,)(pg, base, vnum, data);
}

void test_svst1_vnum_f32(svbool_t pg, float32_t *base, int64_t vnum, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_vnum_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %[[GEP]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_vnum,_f32,,)(pg, base, vnum, data);
}

void test_svst1_vnum_f64(svbool_t pg, float64_t *base, int64_t vnum, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_vnum_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %[[GEP]])
  return SVE_ACLE_FUNC(svst1_vnum,_f64,,)(pg, base, vnum, data);
}

void test_svst1_scatter_u32base_s32(svbool_t pg, svuint32_t bases, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,,_s32)(pg, bases, data);
}

void test_svst1_scatter_u64base_s64(svbool_t pg, svuint64_t bases, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,,_s64)(pg, bases, data);
}

void test_svst1_scatter_u32base_u32(svbool_t pg, svuint32_t bases, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,,_u32)(pg, bases, data);
}

void test_svst1_scatter_u64base_u64(svbool_t pg, svuint64_t bases, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,,_u64)(pg, bases, data);
}

void test_svst1_scatter_u32base_f32(svbool_t pg, svuint32_t bases, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,,_f32)(pg, bases, data);
}

void test_svst1_scatter_u64base_f64(svbool_t pg, svuint64_t bases, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,,_f64)(pg, bases, data);
}

void test_svst1_scatter_s32offset_s32(svbool_t pg, int32_t *base, svint32_t offsets, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s32offset_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s32,offset,_s32)(pg, base, offsets, data);
}

void test_svst1_scatter_s64offset_s64(svbool_t pg, int64_t *base, svint64_t offsets, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s64offset_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s64,offset,_s64)(pg, base, offsets, data);
}

void test_svst1_scatter_s32offset_u32(svbool_t pg, uint32_t *base, svint32_t offsets, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s32offset_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s32,offset,_u32)(pg, base, offsets, data);
}

void test_svst1_scatter_s64offset_u64(svbool_t pg, uint64_t *base, svint64_t offsets, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s64offset_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s64,offset,_u64)(pg, base, offsets, data);
}

void test_svst1_scatter_s32offset_f32(svbool_t pg, float32_t *base, svint32_t offsets, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s32offset_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s32,offset,_f32)(pg, base, offsets, data);
}

void test_svst1_scatter_s64offset_f64(svbool_t pg, float64_t *base, svint64_t offsets, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s64offset_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s64,offset,_f64)(pg, base, offsets, data);
}

void test_svst1_scatter_u32offset_s32(svbool_t pg, int32_t *base, svuint32_t offsets, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32offset_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u32,offset,_s32)(pg, base, offsets, data);
}

void test_svst1_scatter_u64offset_s64(svbool_t pg, int64_t *base, svuint64_t offsets, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64offset_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u64,offset,_s64)(pg, base, offsets, data);
}

void test_svst1_scatter_u32offset_u32(svbool_t pg, uint32_t *base, svuint32_t offsets, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32offset_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u32,offset,_u32)(pg, base, offsets, data);
}

void test_svst1_scatter_u64offset_u64(svbool_t pg, uint64_t *base, svuint64_t offsets, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64offset_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u64,offset,_u64)(pg, base, offsets, data);
}

void test_svst1_scatter_u32offset_f32(svbool_t pg, float32_t *base, svuint32_t offsets, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32offset_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u32,offset,_f32)(pg, base, offsets, data);
}

void test_svst1_scatter_u64offset_f64(svbool_t pg, float64_t *base, svuint64_t offsets, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64offset_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u64,offset,_f64)(pg, base, offsets, data);
}

void test_svst1_scatter_u32base_offset_s32(svbool_t pg, svuint32_t bases, int64_t offset, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_offset_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,_offset,_s32)(pg, bases, offset, data);
}

void test_svst1_scatter_u64base_offset_s64(svbool_t pg, svuint64_t bases, int64_t offset, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_offset_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,_offset,_s64)(pg, bases, offset, data);
}

void test_svst1_scatter_u32base_offset_u32(svbool_t pg, svuint32_t bases, int64_t offset, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_offset_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,_offset,_u32)(pg, bases, offset, data);
}

void test_svst1_scatter_u64base_offset_u64(svbool_t pg, svuint64_t bases, int64_t offset, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_offset_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,_offset,_u64)(pg, bases, offset, data);
}

void test_svst1_scatter_u32base_offset_f32(svbool_t pg, svuint32_t bases, int64_t offset, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_offset_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,_offset,_f32)(pg, bases, offset, data);
}

void test_svst1_scatter_u64base_offset_f64(svbool_t pg, svuint64_t bases, int64_t offset, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_offset_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,_offset,_f64)(pg, bases, offset, data);
}

void test_svst1_scatter_s32index_s32(svbool_t pg, int32_t *base, svint32_t indices, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s32index_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s32,index,_s32)(pg, base, indices, data);
}

void test_svst1_scatter_s64index_s64(svbool_t pg, int64_t *base, svint64_t indices, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s64index_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s64,index,_s64)(pg, base, indices, data);
}

void test_svst1_scatter_s32index_u32(svbool_t pg, uint32_t *base, svint32_t indices, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s32index_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s32,index,_u32)(pg, base, indices, data);
}

void test_svst1_scatter_s64index_u64(svbool_t pg, uint64_t *base, svint64_t indices, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s64index_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s64,index,_u64)(pg, base, indices, data);
}

void test_svst1_scatter_s32index_f32(svbool_t pg, float32_t *base, svint32_t indices, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s32index_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.sxtw.index.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s32,index,_f32)(pg, base, indices, data);
}

void test_svst1_scatter_s64index_f64(svbool_t pg, float64_t *base, svint64_t indices, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_s64index_f64
  // CHECK: %[[PG:.*]]  = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,s64,index,_f64)(pg, base, indices, data);
}

void test_svst1_scatter_u32index_s32(svbool_t pg, int32_t *base, svuint32_t indices, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32index_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u32,index,_s32)(pg, base, indices, data);
}

void test_svst1_scatter_u64index_s64(svbool_t pg, int64_t *base, svuint64_t indices, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64index_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %0, i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u64,index,_s64)(pg, base, indices, data);
}

void test_svst1_scatter_u32index_u32(svbool_t pg, uint32_t *base, svuint32_t indices, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32index_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u32,index,_u32)(pg, base, indices, data);
}

void test_svst1_scatter_u64index_u64(svbool_t pg, uint64_t *base, svuint64_t indices, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64index_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u64,index,_u64)(pg, base, indices, data);
}

void test_svst1_scatter_u32index_f32(svbool_t pg, float32_t *base, svuint32_t indices, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32index_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.uxtw.index.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u32,index,_f32)(pg, base, indices, data);
}

void test_svst1_scatter_u64index_f64(svbool_t pg, float64_t *base, svuint64_t indices, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64index_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.index.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter_,u64,index,_f64)(pg, base, indices, data);
}

void test_svst1_scatter_u32base_index_s32(svbool_t pg, svuint32_t bases, int64_t index, svint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_index_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,_index,_s32)(pg, bases, index, data);
}

void test_svst1_scatter_u64base_index_s64(svbool_t pg, svuint64_t bases, int64_t index, svint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_index_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 3
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,_index,_s64)(pg, bases, index, data);
}

void test_svst1_scatter_u32base_index_u32(svbool_t pg, svuint32_t bases, int64_t index, svuint32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_index_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,_index,_u32)(pg, bases, index, data);
}

void test_svst1_scatter_u64base_index_u64(svbool_t pg, svuint64_t bases, int64_t index, svuint64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_index_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 3
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,_index,_u64)(pg, bases, index, data);
}

void test_svst1_scatter_u32base_index_f32(svbool_t pg, svuint32_t bases, int64_t index, svfloat32_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u32base_index_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 2
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u32base,_index,_f32)(pg, bases, index, data);
}

void test_svst1_scatter_u64base_index_f64(svbool_t pg, svuint64_t bases, int64_t index, svfloat64_t data)
{
  // CHECK-LABEL: test_svst1_scatter_u64base_index_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 3
  // CHECK: call void @llvm.aarch64.sve.st1.scatter.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1_scatter,_u64base,_index,_f64)(pg, bases, index, data);
}
