// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr10 \
// RUN: -emit-llvm %s -o - | FileCheck %s

unsigned long long ulla, ullb;

unsigned long long test_pdepd(void) {
  // CHECK: @llvm.ppc.pdepd
  return __builtin_pdepd(ulla, ullb);
}

unsigned long long test_pextd(void) {
  // CHECK: @llvm.ppc.pextd
  return __builtin_pextd(ulla, ullb);
}
