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

void test_svstnt1_s8(svbool_t pg, int8_t *base, svint8_t data)
{
  // CHECK-LABEL: test_svstnt1_s8
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_s8,,)(pg, base, data);
}

void test_svstnt1_s16(svbool_t pg, int16_t *base, svint16_t data)
{
  // CHECK-LABEL: test_svstnt1_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_s16,,)(pg, base, data);
}

void test_svstnt1_s32(svbool_t pg, int32_t *base, svint32_t data)
{
  // CHECK-LABEL: test_svstnt1_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_s32,,)(pg, base, data);
}

void test_svstnt1_s64(svbool_t pg, int64_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svstnt1_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_s64,,)(pg, base, data);
}

void test_svstnt1_u8(svbool_t pg, uint8_t *base, svuint8_t data)
{
  // CHECK-LABEL: test_svstnt1_u8
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_u8,,)(pg, base, data);
}

void test_svstnt1_u16(svbool_t pg, uint16_t *base, svuint16_t data)
{
  // CHECK-LABEL: test_svstnt1_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_u16,,)(pg, base, data);
}

void test_svstnt1_u32(svbool_t pg, uint32_t *base, svuint32_t data)
{
  // CHECK-LABEL: test_svstnt1_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_u32,,)(pg, base, data);
}

void test_svstnt1_u64(svbool_t pg, uint64_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svstnt1_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_u64,,)(pg, base, data);
}

void test_svstnt1_f16(svbool_t pg, float16_t *base, svfloat16_t data)
{
  // CHECK-LABEL: test_svstnt1_f16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %[[PG]], half* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_f16,,)(pg, base, data);
}

void test_svstnt1_f32(svbool_t pg, float32_t *base, svfloat32_t data)
{
  // CHECK-LABEL: test_svstnt1_f32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_f32,,)(pg, base, data);
}

void test_svstnt1_f64(svbool_t pg, float64_t *base, svfloat64_t data)
{
  // CHECK-LABEL: test_svstnt1_f64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1,_f64,,)(pg, base, data);
}

void test_svstnt1_vnum_s8(svbool_t pg, int8_t *base, int64_t vnum, svint8_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_s8
  // CHECK: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_s8,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_s16(svbool_t pg, int16_t *base, int64_t vnum, svint16_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_s16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_s16,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_s32(svbool_t pg, int32_t *base, int64_t vnum, svint32_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_s32
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_s32,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_s64(svbool_t pg, int64_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_s64
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_u8(svbool_t pg, uint8_t *base, int64_t vnum, svuint8_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_u8
  // CHECK: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_u8,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_u16(svbool_t pg, uint16_t *base, int64_t vnum, svuint16_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_u16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_u16,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_u32(svbool_t pg, uint32_t *base, int64_t vnum, svuint32_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_u32
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_u32,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_u64(svbool_t pg, uint64_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_u64
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_u64,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_f16(svbool_t pg, float16_t *base, int64_t vnum, svfloat16_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_f16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %[[PG]], half* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_f16,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_f32(svbool_t pg, float32_t *base, int64_t vnum, svfloat32_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_f32
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %[[PG]], float* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_f32,,)(pg, base, vnum, data);
}

void test_svstnt1_vnum_f64(svbool_t pg, float64_t *base, int64_t vnum, svfloat64_t data)
{
  // CHECK-LABEL: test_svstnt1_vnum_f64
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %[[PG]], double* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svstnt1_vnum,_f64,,)(pg, base, vnum, data);
}
