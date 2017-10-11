// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:            -o - | FileCheck %s

#include <altivec.h>

// CHECK-LABEL: @_Z5test1Dv8_tS_
// CHECK: @llvm.ppc.altivec.vcmpequh.p
bool test1(vector unsigned short v1, vector unsigned short v2) {
  return v1 == v2;
}

