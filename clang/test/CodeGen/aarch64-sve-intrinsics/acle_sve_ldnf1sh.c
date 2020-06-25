// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

svint32_t test_svldnf1sh_s32(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svldnf1sh_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnf1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldnf1sh_s32(pg, base);
}

svint64_t test_svldnf1sh_s64(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svldnf1sh_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnf1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldnf1sh_s64(pg, base);
}

svuint32_t test_svldnf1sh_u32(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svldnf1sh_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnf1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldnf1sh_u32(pg, base);
}

svuint64_t test_svldnf1sh_u64(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svldnf1sh_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnf1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldnf1sh_u64(pg, base);
}

svint32_t test_svldnf1sh_vnum_s32(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldnf1sh_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnf1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldnf1sh_vnum_s32(pg, base, vnum);
}

svint64_t test_svldnf1sh_vnum_s64(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldnf1sh_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnf1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldnf1sh_vnum_s64(pg, base, vnum);
}

svuint32_t test_svldnf1sh_vnum_u32(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldnf1sh_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnf1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[SEXT]]
  return svldnf1sh_vnum_u32(pg, base, vnum);
}

svuint64_t test_svldnf1sh_vnum_u64(svbool_t pg, const int16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svldnf1sh_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ldnf1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svldnf1sh_vnum_u64(pg, base, vnum);
}
