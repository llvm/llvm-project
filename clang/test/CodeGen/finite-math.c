// RUN: %clang_cc1 -ffinite-math-only -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FINITE
// RUN: %clang_cc1 -fno-signed-zeros -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK  -check-prefix=NSZ
// RUN: %clang_cc1 -freciprocal-math -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK  -check-prefix=RECIP
// RUN: %clang_cc1 -mreassociate -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK  -check-prefix=REASSOC

float f0, f1, f2;

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()

  // FINITE: fadd nnan ninf
  // NSZ: fadd nsz
  // RECIP: fadd arcp
  // REASSOC: fadd reassoc
  f0 = f1 + f2;

  // CHECK: ret
}

