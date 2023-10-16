// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fcx-limited-range -o - | FileCheck %s

_Complex float f1(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @f1
  // CHECK: fmul
  // CHECK-NEXT: fmul
  // CHECK-NEXT: fsub
  // CHECK-NEXT: fmul
  // CHECK-NEXT: fmul
  // CHECK-NEXT: fadd
  return a * b;
}

_Complex float f2(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @f2
  // CHECK: fdiv float
  // CHECK-NEXT: fdiv float
  return a / b;
}
