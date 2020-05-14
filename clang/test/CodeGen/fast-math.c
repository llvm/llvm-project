// RUN: %clang_cc1 -ffast-math -emit-llvm -o - %s | FileCheck %s
float f0, f1, f2;

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()

  // CHECK: fadd fast
  f0 = f1 + f2;

  // CHECK: ret
}
