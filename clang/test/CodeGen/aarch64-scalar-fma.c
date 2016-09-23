// RUN: %clang_cc1 -triple=aarch64-unknown -Os -ffp-contract=fast -S -o - %s | FileCheck -check-prefix=CHECK-FAST -check-prefix=CHECK-ALL %s
// RUN: %clang_cc1 -triple=aarch64-unknown -Os -ffp-contract=on -S -o - %s | FileCheck -check-prefix=CHECK-ON -check-prefix=CHECK-ALL %s
// RUN: %clang_cc1 -triple=aarch64-unknown -Os -ffp-contract=off -S -o - %s | FileCheck -check-prefix=CHECK-OFF -check-prefix=CHECK-ALL %s
// RUN: %clang_cc1 -triple=aarch64-unknown -Os -S -o - %s | FileCheck -check-prefix=CHECK-ON -check-prefix=CHECK-ALL %s
// REQUIRES: aarch64-registered-target

float test1(float x, float y, float z) {
  return x*y + z;
  // CHECK-ALL-LABEL: test1:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test2(double x, double y, double z) {
  z -= x*y;
  return z;
  // CHECK-ALL-LABEL: test2:
  // CHECK-FAST: fmsub
  // CHECK-ON: fmsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

float test3(float x, float y, float z) {
  float tmp = x*y;
  return tmp + z;
  // CHECK-ALL-LABEL: test3:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test4(double x, double y, double z) {
  double tmp = x*y;
  return tmp - z;
  // CHECK-ALL-LABEL: test4:
  // CHECK-FAST: fnmsub
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

#pragma STDC FP_CONTRACT ON

float test5(float x, float y, float z) {
  return x*y + z;
  // CHECK-ALL-LABEL: test5:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test6(double x, double y, double z) {
  z -= x*y;
  return z;
  // CHECK-ALL-LABEL: test6:
  // CHECK-FAST: fmsub
  // CHECK-ON: fmsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

float test7(float x, float y, float z) {
  float tmp = x*y;
  return tmp + z;
  // CHECK-ALL-LABEL: test7:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test8(double x, double y, double z) {
  double tmp = x*y;
  return tmp - z;
  // CHECK-ALL-LABEL: test8:
  // CHECK-FAST: fnmsub
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

#pragma STDC FP_CONTRACT OFF

float test9(float x, float y, float z) {
  return x*y + z;
  // CHECK-ALL-LABEL: test9:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test10(double x, double y, double z) {
  z -= x*y;
  return z;
  // CHECK-ALL-LABEL: test10:
  // CHECK-FAST: fmsub
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

float test11(float x, float y, float z) {
  float tmp = x*y;
  return tmp + z;
  // CHECK-ALL-LABEL: test11:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test12(double x, double y, double z) {
  double tmp = x*y;
  return tmp - z;
  // CHECK-ALL-LABEL: test12:
  // CHECK-FAST: fnmsub
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

#pragma STDC FP_CONTRACT DEFAULT

float test17(float x, float y, float z) {
  return x*y + z;
  // CHECK-ALL-LABEL: test17:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test18(double x, double y, double z) {
  z -= x*y;
  return z;
  // CHECK-ALL-LABEL: test18:
  // CHECK-FAST: fmsub
  // CHECK-ON: fmsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}

float test19(float x, float y, float z) {
  float tmp = x*y;
  return tmp + z;
  // CHECK-ALL-LABEL: test19:
  // CHECK-FAST: fmadd
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fadd
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fadd
}

double test20(double x, double y, double z) {
  double tmp = x*y;
  return tmp - z;
  // CHECK-ALL-LABEL: test20:
  // CHECK-FAST: fnmsub
  // CHECK-ON: fmul
  // CHECK-ON-NEXT: fsub
  // CHECK-OFF: fmul
  // CHECK-OFF-NEXT: fsub
}
