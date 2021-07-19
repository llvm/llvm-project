// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR9-ERR

extern unsigned int ui;

int test_builtin_ppc_cmprb() {
  // CHECK-LABEL: @test_builtin_ppc_cmprb(
  // CHECK:       %2 = call i32 @llvm.ppc.cmprb(i32 0, i32 %0, i32 %1)
  // CHECK:       %5 = call i32 @llvm.ppc.cmprb(i32 1, i32 %3, i32 %4)
  // CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __builtin_ppc_cmprb(0, ui, ui) + __builtin_ppc_cmprb(1, ui, ui);
}

unsigned int extract_exp (double d) {
// CHECK-LABEL: @extract_exp
// CHECK:    [[TMP1:%.*]] = call i32 @llvm.ppc.extract.exp(double %0)
// CHECK-NEXT:    ret i32 [[TMP1]]
// CHECK-NONPWR9-ERR:  error: this builtin is only valid on POWER9 or later CPUs
  return __extract_exp (d);
}
