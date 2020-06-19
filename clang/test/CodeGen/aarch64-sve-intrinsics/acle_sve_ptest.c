// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
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
