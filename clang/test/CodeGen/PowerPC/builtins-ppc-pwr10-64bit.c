// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr10 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr10 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr10 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-32-ERROR
// RUN: not %clang_cc1 -triple powerpc-unknown-linux-gnu -emit-llvm-only %s \
// RUN:   -target-cpu pwr9 2>&1 | FileCheck %s --check-prefix=CHECK-32-ERROR
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr9 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR10-ERR
// RUN: not %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR10-ERR

extern unsigned long long ull;

unsigned long long test_builtin_pextd() {
  // CHECK-LABEL:    @test_builtin_pextd(
  // CHECK:          %2 = call i64 @llvm.ppc.pextd(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR10-ERR:  error: this builtin is only valid on POWER10 or later CPUs
  return __builtin_pextd(ull, ull);
}

unsigned long long test_builtin_pdepd() {
  // CHECK-LABEL:    @test_builtin_pdepd(
  // CHECK:          %2 = call i64 @llvm.ppc.pdepd(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR10-ERR:  error: this builtin is only valid on POWER10 or later CPUs
  return __builtin_pdepd(ull, ull);
}

