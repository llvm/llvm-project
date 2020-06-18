// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +vsx -target-feature +altivec \
// RUN:   -target-cpu pwr10 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <altivec.h>

vector unsigned long long vulla, vullb;

vector unsigned long long test_vpdepd(void) {
  // CHECK: @llvm.ppc.altivec.vpdepd(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_pdep(vulla, vullb);
}

vector unsigned long long test_vpextd(void) {
  // CHECK: @llvm.ppc.altivec.vpextd(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_pext(vulla, vullb);
}
