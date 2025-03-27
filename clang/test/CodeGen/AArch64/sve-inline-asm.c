// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2p1 \
// RUN:   -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2p1 \
// RUN:   -S -o /dev/null

void test_sve_asm(void) {
  asm volatile(
      "ptrue p0.d\n"
      "ptrue p15.d\n"
      "add z0.d, p0/m, z0.d, z0.d\n"
      "add z31.d, p0/m, z31.d, z31.d\n"
      :
      :
      : "z0", "z31", "p0", "p15");
  // CHECK-LABEL: @test_sve_asm
  // CHECK: "~{z0},~{z31},~{p0},~{p15}"
}

void test_sve2p1_asm(void) {
  asm("pfalse pn0.b\n"
      "ptrue pn8.d\n"
      "ptrue pn15.b\n"
      "pext p3.b, pn8[1]\n"
      ::: "pn0", "pn8", "pn15", "p3");
  // CHECK-LABEL: @test_sve2p1_asm
  // CHECK: "~{pn0},~{pn8},~{pn15},~{p3}"
}
