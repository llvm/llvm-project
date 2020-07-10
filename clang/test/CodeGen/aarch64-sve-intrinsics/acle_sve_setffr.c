// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// CHECK-NOT: warning
// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

void test_svsetffr()
{
  // CHECK-LABEL: test_svsetffr
  // CHECK: call void @llvm.aarch64.sve.setffr()
  // CHECK: ret void
  svsetffr();
}
