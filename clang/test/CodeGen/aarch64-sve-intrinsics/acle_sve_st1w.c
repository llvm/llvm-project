// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - -emit-llvm %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svst1w_s64(svbool_t pg, int32_t *base, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK: call void @llvm.masked.store.nxv2i32.p0nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i32>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w,_s64,,)(pg, base, data);
}

void test_svst1w_u64(svbool_t pg, uint32_t *base, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK: call void @llvm.masked.store.nxv2i32.p0nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i32>* %[[BASE]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w,_u64,,)(pg, base, data);
}

void test_svst1w_vnum_s64(svbool_t pg, int32_t *base, int64_t vnum, svint64_t data)
{
  // CHECK-LABEL: test_svst1w_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i32.p0nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i32>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_vnum,_s64,,)(pg, base, vnum, data);
}

void test_svst1w_vnum_u64(svbool_t pg, uint32_t *base, int64_t vnum, svuint64_t data)
{
  // CHECK-LABEL: test_svst1w_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i32* %base to <vscale x 2 x i32>*
  // CHECK-DAG: %[[DATA:.*]] = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %[[BASE]], i64 %vnum
  // CHECK: call void @llvm.masked.store.nxv2i32.p0nxv2i32(<vscale x 2 x i32> %[[DATA]], <vscale x 2 x i32>* %[[GEP]], i32 1, <vscale x 2 x i1> %[[PG]])
  // CHECK: ret void
  return SVE_ACLE_FUNC(svst1w_vnum,_u64,,)(pg, base, vnum, data);
}
