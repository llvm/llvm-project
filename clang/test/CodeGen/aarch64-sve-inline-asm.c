// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK

void test_sve_asm() {
  asm volatile(
      "ptrue p0.d\n"
      "ptrue p15.d\n"
      "add z0.d, p0/m, z0.d, z0.d\n"
      "add z31.d, p0/m, z31.d, z31.d\n"
      :
      :
      : "z0", "z31", "p0", "p15");
  // CHECK: "~{z0},~{z31},~{p0},~{p15}"
}
