// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svst3_bf16(svbool_t pg, bfloat16_t *base, svbfloat16x3_t data)
{
  // CHECK-LABEL: test_svst3_bf16
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %data, i32 1)
  // CHECK-DAG: %[[V2:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %data, i32 2)
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st3.nxv8bf16(<vscale x 8 x bfloat> %[[V0]], <vscale x 8 x bfloat> %[[V1]], <vscale x 8 x bfloat> %[[V2]], <vscale x 8 x i1> %[[PG]], bfloat* %base)
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svst3,_bf16,,)(pg, base, data);
}

void test_svst3_vnum_bf16(svbool_t pg, bfloat16_t *base, int64_t vnum, svbfloat16x3_t data)
{
  // CHECK-LABEL: test_svst3_vnum_bf16
  // CHECK-DAG: %[[BITCAST:.*]] = bitcast bfloat* %base to <vscale x 8 x bfloat>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %[[BITCAST]], i64 %vnum, i64 0
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %data, i32 1)
  // CHECK-DAG: %[[V2:.*]] = call <vscale x 8 x bfloat> @llvm.aarch64.sve.tuple.get.nxv8bf16.nxv24bf16(<vscale x 24 x bfloat> %data, i32 2)
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: call void @llvm.aarch64.sve.st3.nxv8bf16(<vscale x 8 x bfloat> %[[V0]], <vscale x 8 x bfloat> %[[V1]], <vscale x 8 x bfloat> %[[V2]], <vscale x 8 x i1> %[[PG]], bfloat* %[[GEP]])
  // CHECK-NEXT: ret
  return SVE_ACLE_FUNC(svst3_vnum,_bf16,,)(pg, base, vnum, data);
}
