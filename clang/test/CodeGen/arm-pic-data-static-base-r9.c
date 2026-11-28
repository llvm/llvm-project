// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -O2 \
// RUN:   -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative \
// RUN:   -S -o - %s | FileCheck %s

// r9 is the incoming static-base register under
// -mno-pic-data-is-text-relative, so ordinary register allocation must not use
// it even in a register-pressure-heavy function that does not reference data.

int pressure(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7,
             int a8, int a9, int a10, int a11) {
  int s0 = a0 + a1;
  int s1 = a2 + a3;
  int s2 = a4 + a5;
  int s3 = a6 + a7;
  int s4 = a8 + a9;
  int s5 = a10 + a11;
  int p0 = s0 * s3;
  int p1 = s1 * s4;
  int p2 = s2 * s5;
  return (p0 ^ p1) + p2 + s0 + s1 + s2 + s3 + s4 + s5;
}

// CHECK-LABEL: pressure:
// CHECK-NOT: r9
// CHECK: .fnend
