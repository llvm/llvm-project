// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint16_t test_svldff1sb_s16(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1sb_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i8> @llvm.aarch64.sve.ldff1.nxv8i8(<vscale x 8 x i1> %[[PG]], i8* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 8 x i8> %[[LOAD]] to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[SEXT]]
  return svldff1sb_s16(pg, base);
}

svint32_t test_svldff1sb_s32(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1sb_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.nxv4i8(<vscale x 4 x i1> %[[PG]], i8* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i8> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldff1sb_s32(pg, base);
}

svint64_t test_svldff1sb_s64(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1sb_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.nxv2i8(<vscale x 2 x i1> %[[PG]], i8* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i8> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldff1sb_s64(pg, base);
}

svuint16_t test_svldff1sb_u16(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1sb_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i8> @llvm.aarch64.sve.ldff1.nxv8i8(<vscale x 8 x i1> %[[PG]], i8* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 8 x i8> %[[LOAD]] to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[SEXT]]
  return svldff1sb_u16(pg, base);
}

svuint32_t test_svldff1sb_u32(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1sb_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.nxv4i8(<vscale x 4 x i1> %[[PG]], i8* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i8> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldff1sb_u32(pg, base);
}

svuint64_t test_svldff1sb_u64(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svldff1sb_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.nxv2i8(<vscale x 2 x i1> %[[PG]], i8* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i8> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldff1sb_u64(pg, base);
}

svint16_t test_svldff1sb_vnum_s16(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1sb_vnum_s16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i8> @llvm.aarch64.sve.ldff1.nxv8i8(<vscale x 8 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 8 x i8> %[[LOAD]] to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[SEXT]]
  return svldff1sb_vnum_s16(pg, base, vnum);
}

svint32_t test_svldff1sb_vnum_s32(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1sb_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.nxv4i8(<vscale x 4 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i8> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldff1sb_vnum_s32(pg, base, vnum);
}

svint64_t test_svldff1sb_vnum_s64(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1sb_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.nxv2i8(<vscale x 2 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i8> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldff1sb_vnum_s64(pg, base, vnum);
}

svuint16_t test_svldff1sb_vnum_u16(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1sb_vnum_u16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 8 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i8> @llvm.aarch64.sve.ldff1.nxv8i8(<vscale x 8 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 8 x i8> %[[LOAD]] to <vscale x 8 x i16>
  // CHECK: ret <vscale x 8 x i16> %[[SEXT]]
  return svldff1sb_vnum_u16(pg, base, vnum);
}

svuint32_t test_svldff1sb_vnum_u32(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1sb_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 4 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.nxv4i8(<vscale x 4 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i8> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldff1sb_vnum_u32(pg, base, vnum);
}

svuint64_t test_svldff1sb_vnum_u64(svbool_t pg, const int8_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldff1sb_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i8* %base to <vscale x 2 x i8>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.nxv2i8(<vscale x 2 x i1> %[[PG]], i8* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i8> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldff1sb_vnum_u64(pg, base, vnum);
}
