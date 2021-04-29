// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null
#include <arm_sve.h>

bool test_svptest_any(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svptest_any
  // CHECK: %[[INTRINSIC:.*]] = call i1 @llvm.aarch64.sve.ptest.any{{(.nxv16i1)?}}(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %op)
  // CHECK: ret i1 %[[INTRINSIC]]
  return svptest_any(pg, op);
}

bool test_svptest_first(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svptest_first
  // CHECK: %[[INTRINSIC:.*]] = call i1 @llvm.aarch64.sve.ptest.first{{(.nxv16i1)?}}(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %op)
  // CHECK: ret i1 %[[INTRINSIC]]
  return svptest_first(pg, op);
}

bool test_svptest_last(svbool_t pg, svbool_t op)
{
  // CHECK-LABEL: test_svptest_last
  // CHECK: %[[INTRINSIC:.*]] = call i1 @llvm.aarch64.sve.ptest.last{{(.nxv16i1)?}}(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %op)
  // CHECK: ret i1 %[[INTRINSIC]]
  return svptest_last(pg, op);
}
