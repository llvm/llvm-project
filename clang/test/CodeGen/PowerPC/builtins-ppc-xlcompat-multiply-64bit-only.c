// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 2>&1 | FileCheck %s --check-prefix=CHECK-32-ERROR

extern long int sli;
extern unsigned long int uli;

long long test_builtin_ppc_mulhd() {
  // CHECK-LABEL:    @test_builtin_ppc_mulhd(
  // CHECK:          %2 = call i64 @llvm.ppc.mulhd(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  return __builtin_ppc_mulhd(sli, sli);
}

unsigned long long test_builtin_ppc_mulhdu() {
  // CHECK-LABEL:    @test_builtin_ppc_mulhdu(
  // CHECK:          %2 = call i64 @llvm.ppc.mulhdu(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  return __builtin_ppc_mulhdu(uli, uli);
}
