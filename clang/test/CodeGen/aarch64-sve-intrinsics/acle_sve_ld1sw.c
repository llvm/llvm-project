// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint64_t test_svld1sw_s64(svbool_t pg, const int32_t *base)
{
  // CHECK-LABEL: test_svld1sw_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i32> zeroinitializer)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i32> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svld1sw_s64(pg, base);
}

svuint64_t test_svld1sw_u64(svbool_t pg, const int32_t *base)
{
  // CHECK-LABEL: test_svld1sw_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i32> zeroinitializer)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i32> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svld1sw_u64(pg, base);
}

svint64_t test_svld1sw_vnum_s64(svbool_t pg, const int32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1sw_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i32> zeroinitializer)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i32> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svld1sw_vnum_s64(pg, base, vnum);
}

svuint64_t test_svld1sw_vnum_u64(svbool_t pg, const int32_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1sw_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %[[BASE]], i64 %vnum
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]], <vscale x 2 x i32> zeroinitializer)
  // CHECK: %[[SEXT:.*]] = sext <vscale x 2 x i32> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[SEXT]]
  return svld1sw_vnum_u64(pg, base, vnum);
}
