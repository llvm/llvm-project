// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_on_mul
  // LIMITED: %[[AC:[^ ]+]] = fmul
  // LIMITED-NEXT: %[[BD:[^ ]+]] = fmul
  // LIMITED-NEXT: %[[RR:[^ ]+]] = fsub
  // LIMITED-NEXT: %[[BC:[^ ]+]] = fmul
  // LIMITED-NEXT: %[[AD:[^ ]+]] = fmul
  // LIMITED-NEXT: %[[RI:[^ ]+]] = fadd
  // LIMITED: ret
  return a * b;
}

_Complex float mul(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @mul
  // FULL: call {{.*}} @__mulsc3(
  // FULL: ret
  return a * b;
}
