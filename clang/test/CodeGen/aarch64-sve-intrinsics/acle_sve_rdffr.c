// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

svbool_t test_svrdffr()
{
  // CHECK-LABEL: test_svrdffr
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.rdffr()
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svrdffr();
}

svbool_t test_svrdffr_z(svbool_t pg)
{
  // CHECK-LABEL: test_svrdffr_z
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.rdffr.z(<vscale x 16 x i1> %pg)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svrdffr_z(pg);
}
