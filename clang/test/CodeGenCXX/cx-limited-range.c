// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fcx-limited-range -o - | FileCheck %s
// RUN %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -fuse-complex-intrinsics -fcx-range=nonan -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-NO-NAN
// RUN %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -fuse-complex-intrinsics -fcx-range=full -o - | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-FULL

_Complex float f1(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // CHECK: fdiv float
  // CHECK: fdiv float
  return a / b;
}

_Complex float f2(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // CHECK: fdiv float
  // CHECK: fdiv float
  return a * b;
}
