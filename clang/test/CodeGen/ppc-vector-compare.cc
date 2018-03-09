// RUN: %clang_cc1 -target-feature +vsx -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:            -o - | FileCheck %s

#include <altivec.h>

// CHECK-LABEL: @_Z5test1Dv8_tS_
// CHECK: @llvm.ppc.altivec.vcmpequh.p
bool test1(vector unsigned short v1, vector unsigned short v2) {
  return v1 == v2;
}

// CHECK-LABEL: @_Z5test2Dv2_mS_Dv2_lS0_Dv2_yS1_Dv2_xS2_Dv2_dS3_
bool test2(vector unsigned long v1, vector unsigned long v2,
           vector long v3, vector long v4,
           vector unsigned long long v5, vector unsigned long long v6,
           vector long long v7, vector long long v8,
           vector double v9, vector double v10) {
  // CHECK: @llvm.ppc.altivec.vcmpequd.p
  bool res = v1 == v2;

  // CHECK: @llvm.ppc.altivec.vcmpequd.p
  res |= v3 == v4;

  // CHECK: @llvm.ppc.altivec.vcmpequd.p
  res |= v5 == v6;

  // CHECK: @llvm.ppc.altivec.vcmpequd.p
  res |= v7 == v8;

  // CHECK: @llvm.ppc.vsx.xvcmpeqdp.p
  res |= v9 == v10;
  return res;
}

