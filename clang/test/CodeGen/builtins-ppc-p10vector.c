// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +vsx -target-feature +altivec \
// RUN:   -target-cpu pwr10 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <altivec.h>

vector signed char vsca;
vector unsigned char vuca;
vector unsigned long long vulla, vullb;
unsigned int uia;

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

vector signed char test_vec_vclrl_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrl(vsca, uia);
}

vector unsigned char test_vec_clrl_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrl(vuca, uia);
}

vector signed char test_vec_vclrr_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrr(vsca, uia);
}

vector unsigned char test_vec_clrr_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrr(vuca, uia);
}
