// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple=x86_64-unknown-linux -S %s -faddrsig -O -o - | FileCheck --check-prefix=ADDRSIG %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -S %s -O -o - | FileCheck --check-prefix=NO-ADDRSIG %s

// ADDRSIG: .addrsig
// ADDRSIG: .addrsig_sym g1
// ADDRSIG-NOT: .addrsig_sym g2

// NO-ADDRSIG-NOT: .addrsig

extern const int g1[], g2[];

const int *f1() {
  return g1;
}

int f2() {
  return g2[0];
}
