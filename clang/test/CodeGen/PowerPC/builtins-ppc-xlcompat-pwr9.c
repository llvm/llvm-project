// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix %s -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR9-ERR

extern unsigned int ui;

int test_builtin_ppc_cmprb_extract_exp(double d) {
  // CHECK-LABEL: @test_builtin_ppc_cmprb_extract_exp(
  // CHECK:       %2 = call i32 @llvm.ppc.cmprb(i32 0, i32 %0, i32 %1)
  // CHECK:       %5 = call i32 @llvm.ppc.cmprb(i32 1, i32 %3, i32 %4)
  // CHECK:       %7 = call i32 @llvm.ppc.extract.exp(double %6)
  // CHECK-NONPWR9-ERR:  error: '__builtin_ppc_cmprb' needs target feature isa-v30-instructions
  // CHECK-NONPWR9-ERR:  error: '__builtin_ppc_extract_exp' needs target feature power9-vector
  return __builtin_ppc_cmprb(0, ui, ui) + __builtin_ppc_cmprb(1, ui, ui) + __extract_exp(d);
}
