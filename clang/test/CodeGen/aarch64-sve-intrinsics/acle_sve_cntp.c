// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

uint64_t test_svcntp_b8(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svcntp_b8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntp.nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %op)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntp_b8(pg, op);
}

uint64_t test_svcntp_b16(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svcntp_b16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[OP:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %op)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntp.nxv8i1(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i1> %[[OP]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntp_b16(pg, op);
}

uint64_t test_svcntp_b32(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svcntp_b32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[OP:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %op)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntp.nxv4i1(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i1> %[[OP]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntp_b32(pg, op);
}

uint64_t test_svcntp_b64(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svcntp_b64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[OP:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %op)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntp.nxv2i1(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i1> %[[OP]])
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntp_b64(pg, op);
}
