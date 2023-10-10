// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s

_Complex float pragma_on_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @pragma_on_mul
  // CHECK: %[[AC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[BD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[RR:[^ ]+]] = fsub
  // CHECK-NEXT: %[[BC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[AD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[RI:[^ ]+]] = fadd
  // CHECK: ret
  return a * b;
}

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_mul
  // CHECK: %[[AC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[BD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[AD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[BC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[RR:[^ ]+]] = fsub
  // CHECK-NEXT: %[[RI:[^ ]+]] = fadd
  // CHECK: ret
  return a * b;
}

_Complex float no_pragma_mul(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @pragma_off_mul
  // CHECK: %[[AC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[BD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[AD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[BC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[RR:[^ ]+]] = fsub
  // CHECK-NEXT: %[[RI:[^ ]+]] = fadd
  // CHECK: ret
  return a * b;
}

_Complex float pragma_on_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @pragma_on_div
  // CHECK: [[AC:%.*]] = fmul
  // CHECK: [[BD:%.*]] = fmul
  // CHECK: [[ACpBD:%.*]] = fadd
  // CHECK: [[CC:%.*]] = fmul
  // CHECK: [[DD:%.*]] = fmul
  // CHECK: [[CCpDD:%.*]] = fadd
  // CHECK: [[BC:%.*]] = fmul
  // CHECK: [[AD:%.*]] = fmul
  // CHECK: [[BCmAD:%.*]] = fsub
  // CHECK: fdiv
  // CHECK: fdiv
  // CHECK: ret
  return a / b;
}

_Complex float pragma_off_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_div
  // CHECK: call {{.*}} @__divsc3(
  // CHECK: ret
  return a / b;
}

_Complex float no_pragma_div(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @no_prama_div
  // CHECK: call {{.*}} @__divsc3(
  // CHECK: ret
  return a / b;
}
