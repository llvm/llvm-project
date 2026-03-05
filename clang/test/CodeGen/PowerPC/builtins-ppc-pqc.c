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

vector unsigned int vuia, vuib;
vector unsigned short vusa, vusb;
vector signed int vsia, vsib;
vector signed short vssa, vssb;

// Test vec_add for unsigned int
vector unsigned int test_vec_add_ui(void) {
  // CHECK-LABEL: @test_vec_add_ui
  // CHECK: add <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_add(vuia, vuib);
}

// Test vec_add for unsigned short
vector unsigned short test_vec_add_uh(void) {
  // CHECK-LABEL: @test_vec_add_uh
  // CHECK: add <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_add(vusa, vusb);
}

// Test vec_sub for unsigned int
vector unsigned int test_vec_sub_ui(void) {
  // CHECK-LABEL: @test_vec_sub_ui
  // CHECK: sub <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_sub(vuia, vuib);
}

// Test vec_sub for unsigned short
vector unsigned short test_vec_sub_uh(void) {
  // CHECK-LABEL: @test_vec_sub_uh
  // CHECK: sub <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_sub(vusa, vusb);
}

// Test vec_mul for unsigned int
vector unsigned int test_vec_mul_ui(void) {
  // CHECK-LABEL: @test_vec_mul_ui
  // CHECK: mul <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_mul(vuia, vuib);
}

// Test vec_mul for unsigned short
vector unsigned short test_vec_mul_uh(void) {
  // CHECK-LABEL: @test_vec_mul_uh
  // CHECK: mul <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_mul(vusa, vusb);
}

// Test vec_mulh for signed int
vector signed int test_vec_mulh_si(void) {
  // CHECK-LABEL: @test_vec_mulh_si
  // CHECK: call <4 x i32> @llvm.ppc.altivec.vmulhsw(<4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_mulh(vsia, vsib);
}

// Test vec_mulh for unsigned int
vector unsigned int test_vec_mulh_ui(void) {
  // CHECK-LABEL: @test_vec_mulh_ui
  // CHECK: call <4 x i32> @llvm.ppc.altivec.vmulhuw(<4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_mulh(vuia, vuib);
}

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

// Made with Bob
