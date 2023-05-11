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

void test_xlcompat() {
  // CHECK-LABEL: @test_xlcompat(
  // CHECK: %2 = call i64 @llvm.ppc.pextd(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR10-ERR: error: '__builtin_pextd' needs target feature isa-v31-instructions
  ull = __builtin_pextd(ull, ull);

  // CHECK: %5 = call i64 @llvm.ppc.pdepd(i64 %3, i64 %4)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR10-ERR: error: '__builtin_pdepd' needs target feature isa-v31-instructions
  ull = __builtin_pdepd(ull, ull);
}
