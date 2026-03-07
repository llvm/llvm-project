// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx \
// RUN:   -target-feature +isa-future-instructions -target-cpu future \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx \
// RUN:   -target-feature +isa-future-instructions -target-cpu future \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <altivec.h>

vector unsigned short vusa, vusb;
vector signed short vssa, vssb;

// Test vec_mulh for signed short
vector signed short test_vec_mulh_ss(void) {
  // CHECK-LABEL: @test_vec_mulh_ss
  // CHECK: call <8 x i16> @llvm.ppc.altivec.vmulhsh(<8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}})
  // CHECK-NEXT: ret <8 x i16>
  return vec_mulh(vssa, vssb);
}

// Test vec_mulh for unsigned short
vector unsigned short test_vec_mulh_uh(void) {
  // CHECK-LABEL: @test_vec_mulh_uh
  // CHECK: call <8 x i16> @llvm.ppc.altivec.vmulhuh(<8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}})
  // CHECK-NEXT: ret <8 x i16>
  return vec_mulh(vusa, vusb);
}
