// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
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

svint8_t test_svldff1_s8(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1_s8
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ldff1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_s8,,)(pg, base);
}

svint16_t test_svldff1_s16(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svldff1_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ldff1.nxv8i16(<vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_s16,,)(pg, base);
}

svint32_t test_svldff1_s32(svbool_t pg, const int32_t *base)
{
  // CHECK-LABEL: test_svldff1_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.nxv4i32(<vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_s32,,)(pg, base);
}

svint64_t test_svldff1_s64(svbool_t pg, const int64_t *base)
{
  // CHECK-LABEL: test_svldff1_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.nxv2i64(<vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_s64,,)(pg, base);
}

svuint8_t test_svldff1_u8(svbool_t pg, const uint8_t *base)
{
  // CHECK-LABEL: test_svldff1_u8
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ldff1.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_u8,,)(pg, base);
}

svuint16_t test_svldff1_u16(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svldff1_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ldff1.nxv8i16(<vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_u16,,)(pg, base);
}

svuint32_t test_svldff1_u32(svbool_t pg, const uint32_t *base)
{
  // CHECK-LABEL: test_svldff1_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.nxv4i32(<vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_u32,,)(pg, base);
}

svuint64_t test_svldff1_u64(svbool_t pg, const uint64_t *base)
{
  // CHECK-LABEL: test_svldff1_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.nxv2i64(<vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_u64,,)(pg, base);
}

svfloat16_t test_svldff1_f16(svbool_t pg, const float16_t *base)
{
  // CHECK-LABEL: test_svldff1_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ldff1.nxv8f16(<vscale x 8 x i1> %[[PG]], half* %base)
  // CHECK: ret <vscale x 8 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_f16,,)(pg, base);
}

svfloat32_t test_svldff1_f32(svbool_t pg, const float32_t *base)
{
  // CHECK-LABEL: test_svldff1_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.nxv4f32(<vscale x 4 x i1> %[[PG]], float* %base)
  // CHECK: ret <vscale x 4 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_f32,,)(pg, base);
}

svfloat64_t test_svldff1_f64(svbool_t pg, const float64_t *base)
{
  // CHECK-LABEL: test_svldff1_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.nxv2f64(<vscale x 2 x i1> %[[PG]], double* %base)
  // CHECK: ret <vscale x 2 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1,_f64,,)(pg, base);
}

svint8_t test_svldff1_vnum_s8(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_s8
  // CHECK: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ldff1.nxv16i8(<vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_s8,,)(pg, base, vnum);
}

svint16_t test_svldff1_vnum_s16(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_s16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ldff1.nxv8i16(<vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_s16,,)(pg, base, vnum);
}

svint32_t test_svldff1_vnum_s32(svbool_t pg, const int32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_s32
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.nxv4i32(<vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_s32,,)(pg, base, vnum);
}

svint64_t test_svldff1_vnum_s64(svbool_t pg, const int64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_s64
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.nxv2i64(<vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_s64,,)(pg, base, vnum);
}

svuint8_t test_svldff1_vnum_u8(svbool_t pg, const uint8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_u8
  // CHECK: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ldff1.nxv16i8(<vscale x 16 x i1> %pg, i8* %[[GEP]])
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_u8,,)(pg, base, vnum);
}

svuint16_t test_svldff1_vnum_u16(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_u16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 8 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ldff1.nxv8i16(<vscale x 8 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_u16,,)(pg, base, vnum);
}

svuint32_t test_svldff1_vnum_u32(svbool_t pg, const uint32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_u32
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i32* %base to <vscale x 4 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.nxv4i32(<vscale x 4 x i1> %[[PG]], i32* %[[GEP]])
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_u32,,)(pg, base, vnum);
}

svuint64_t test_svldff1_vnum_u64(svbool_t pg, const uint64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_u64
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i64* %base to <vscale x 2 x i64>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.nxv2i64(<vscale x 2 x i1> %[[PG]], i64* %[[GEP]])
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_u64,,)(pg, base, vnum);
}

svfloat16_t test_svldff1_vnum_f16(svbool_t pg, const float16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_f16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast half* %base to <vscale x 8 x half>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ldff1.nxv8f16(<vscale x 8 x i1> %[[PG]], half* %[[GEP]])
  // CHECK: ret <vscale x 8 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_f16,,)(pg, base, vnum);
}

svfloat32_t test_svldff1_vnum_f32(svbool_t pg, const float32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_f32
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast float* %base to <vscale x 4 x float>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.nxv4f32(<vscale x 4 x i1> %[[PG]], float* %[[GEP]])
  // CHECK: ret <vscale x 4 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_f32,,)(pg, base, vnum);
}

svfloat64_t test_svldff1_vnum_f64(svbool_t pg, const float64_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1_vnum_f64
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast double* %base to <vscale x 2 x double>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.nxv2f64(<vscale x 2 x i1> %[[PG]], double* %[[GEP]])
  // CHECK: ret <vscale x 2 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svldff1_vnum,_f64,,)(pg, base, vnum);
}

svint32_t test_svldff1_gather_u32base_s32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svldff1_gather_u32base_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _s32, )(pg, bases);
}

svint64_t test_svldff1_gather_u64base_s64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svldff1_gather_u64base_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _s64, )(pg, bases);
}

svuint32_t test_svldff1_gather_u32base_u32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svldff1_gather_u32base_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _u32, )(pg, bases);
}

svuint64_t test_svldff1_gather_u64base_u64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svldff1_gather_u64base_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _u64, )(pg, bases);
}

svfloat32_t test_svldff1_gather_u32base_f32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svldff1_gather_u32base_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _f32, )(pg, bases);
}

svfloat64_t test_svldff1_gather_u64base_f64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svldff1_gather_u64base_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _f64, )(pg, bases);
}

svint32_t test_svldff1_gather_s32offset_s32(svbool_t pg, const int32_t *base, svint32_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_s32offset_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s32, offset, _s32)(pg, base, offsets);
}

svint64_t test_svldff1_gather_s64offset_s64(svbool_t pg, const int64_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_s64offset_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s64, offset, _s64)(pg, base, offsets);
}

svuint32_t test_svldff1_gather_s32offset_u32(svbool_t pg, const uint32_t *base, svint32_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_s32offset_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s32, offset, _u32)(pg, base, offsets);
}

svuint64_t test_svldff1_gather_s64offset_u64(svbool_t pg, const uint64_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_s64offset_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s64, offset, _u64)(pg, base, offsets);
}

svfloat32_t test_svldff1_gather_s32offset_f32(svbool_t pg, const float32_t *base, svint32_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_s32offset_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.sxtw.nxv4f32(<vscale x 4 x i1> [[PG]], float* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s32, offset, _f32)(pg, base, offsets);
}

svfloat64_t test_svldff1_gather_s64offset_f64(svbool_t pg, const float64_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_s64offset_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s64, offset, _f64)(pg, base, offsets);
}

svint32_t test_svldff1_gather_u32offset_s32(svbool_t pg, const int32_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_u32offset_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u32, offset, _s32)(pg, base, offsets);
}

svint64_t test_svldff1_gather_u64offset_s64(svbool_t pg, const int64_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_u64offset_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u64, offset, _s64)(pg, base, offsets);
}

svuint32_t test_svldff1_gather_u32offset_u32(svbool_t pg, const uint32_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_u32offset_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u32, offset, _u32)(pg, base, offsets);
}

svuint64_t test_svldff1_gather_u64offset_u64(svbool_t pg, const uint64_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_u64offset_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u64, offset, _u64)(pg, base, offsets);
}

svfloat32_t test_svldff1_gather_u32offset_f32(svbool_t pg, const float32_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_u32offset_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.uxtw.nxv4f32(<vscale x 4 x i1> [[PG]], float* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u32, offset, _f32)(pg, base, offsets);
}

svfloat64_t test_svldff1_gather_u64offset_f64(svbool_t pg, const float64_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svldff1_gather_u64offset_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u64, offset, _f64)(pg, base, offsets);
}

svint32_t test_svldff1_gather_u32base_offset_s32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldff1_gather_u32base_offset_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _offset_s32, )(pg, bases, offset);
}

svint64_t test_svldff1_gather_u64base_offset_s64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldff1_gather_u64base_offset_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _offset_s64, )(pg, bases, offset);
}

svuint32_t test_svldff1_gather_u32base_offset_u32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldff1_gather_u32base_offset_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _offset_u32, )(pg, bases, offset);
}

svuint64_t test_svldff1_gather_u64base_offset_u64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldff1_gather_u64base_offset_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _offset_u64, )(pg, bases, offset);
}

svfloat32_t test_svldff1_gather_u32base_offset_f32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldff1_gather_u32base_offset_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _offset_f32, )(pg, bases, offset);
}

svfloat64_t test_svldff1_gather_u64base_offset_f64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldff1_gather_u64base_offset_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _offset_f64, )(pg, bases, offset);
}

svint32_t test_svldff1_gather_s32index_s32(svbool_t pg, const int32_t *base, svint32_t indices) {
  // CHECK-LABEL: test_svldff1_gather_s32index_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s32, index, _s32)(pg, base, indices);
}

svint64_t test_svldff1_gather_s64index_s64(svbool_t pg, const int64_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svldff1_gather_s64index_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s64, index, _s64)(pg, base, indices);
}

svuint32_t test_svldff1_gather_s32index_u32(svbool_t pg, const uint32_t *base, svint32_t indices) {
  // CHECK-LABEL: test_svldff1_gather_s32index_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s32, index, _u32)(pg, base, indices);
}

svuint64_t test_svldff1_gather_s64index_u64(svbool_t pg, const uint64_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svldff1_gather_s64index_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s64, index, _u64)(pg, base, indices);
}

svfloat32_t test_svldff1_gather_s32index_f32(svbool_t pg, const float32_t *base, svint32_t indices) {
  // CHECK-LABEL: test_svldff1_gather_s32index_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.sxtw.index.nxv4f32(<vscale x 4 x i1> [[PG]], float* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s32, index, _f32)(pg, base, indices);
}

svfloat64_t test_svldff1_gather_s64index_f64(svbool_t pg, const float64_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svldff1_gather_s64index_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.index.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, s64, index, _f64)(pg, base, indices);
}

svint32_t test_svldff1_gather_u32index_s32(svbool_t pg, const int32_t *base, svuint32_t indices) {
  // CHECK-LABEL: test_svldff1_gather_u32index_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u32, index, _s32)(pg, base, indices);
}

svint64_t test_svldff1_gather_u64index_s64(svbool_t pg, const int64_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svldff1_gather_u64index_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u64, index, _s64)(pg, base, indices);
}

svuint32_t test_svldff1_gather_u32index_u32(svbool_t pg, const uint32_t *base, svuint32_t indices) {
  // CHECK-LABEL: test_svldff1_gather_u32index_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u32, index, _u32)(pg, base, indices);
}

svuint64_t test_svldff1_gather_u64index_u64(svbool_t pg, const uint64_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svldff1_gather_u64index_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u64, index, _u64)(pg, base, indices);
}

svfloat32_t test_svldff1_gather_u32index_f32(svbool_t pg, const float32_t *base, svuint32_t indices) {
  // CHECK-LABEL: test_svldff1_gather_u32index_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.uxtw.index.nxv4f32(<vscale x 4 x i1> [[PG]], float* %base, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u32, index, _f32)(pg, base, indices);
}

svfloat64_t test_svldff1_gather_u64index_f64(svbool_t pg, const float64_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svldff1_gather_u64index_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.index.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather_, u64, index, _f64)(pg, base, indices);
}

svint32_t test_svldff1_gather_u32base_index_s32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svldff1_gather_u32base_index_s32
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 2
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _index_s32, )(pg, bases, index);
}

svint64_t test_svldff1_gather_u64base_index_s64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svldff1_gather_u64base_index_s64
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 3
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _index_s64, )(pg, bases, index);
}

svuint32_t test_svldff1_gather_u32base_index_u32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svldff1_gather_u32base_index_u32
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 2
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _index_u32, )(pg, bases, index);
}

svuint64_t test_svldff1_gather_u64base_index_u64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svldff1_gather_u64base_index_u64
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 3
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _index_u64, )(pg, bases, index);
}

svfloat32_t test_svldff1_gather_u32base_index_f32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svldff1_gather_u32base_index_f32
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 2
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u32base, _index_f32, )(pg, bases, index);
}

svfloat64_t test_svldff1_gather_u64base_index_f64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svldff1_gather_u64base_index_f64
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 3
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  return SVE_ACLE_FUNC(svldff1_gather, _u64base, _index_f64, )(pg, bases, index);
}
