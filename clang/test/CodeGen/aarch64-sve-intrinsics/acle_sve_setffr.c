// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

void test_svsetffr()
{
  // CHECK-LABEL: test_svsetffr
  // CHECK: call void @llvm.aarch64.sve.setffr()
  // CHECK: ret void
  svsetffr();
}
