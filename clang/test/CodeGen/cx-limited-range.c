// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fcx-limited-range -o - | FileCheck %s

// RUN %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -fuse-complex-intrinsics -fcx-range=nonan -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-NO-NAN

// RUN %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -fuse-complex-intrinsics -fcx-range=full -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-FULL

_Complex float f1(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @f1
  // CHECK: fdiv float
  // CHECK-NEXT: fdiv float
  return a / b;
}

_Complex float f2(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @f2
  // CHECK: %[[AC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[BD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[RR:[^ ]+]] = fsub
  // CHECK-NEXT: %[[BC:[^ ]+]] = fmul
  // CHECK-NEXT: %[[AD:[^ ]+]] = fmul
  // CHECK-NEXT: %[[RI:[^ ]+]] = fadd
  // CHECK: ret
  return a * b;
}
