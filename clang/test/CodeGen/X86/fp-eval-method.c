// RUN: %clang_cc1 -triple i386-unknown-netbsd -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK,CHECK-SRC

// RUN: %clang_cc1 -triple i386--linux -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK,CHECK-SRC

// RUN: %clang_cc1 -triple i386--linux -target-feature +x87-excess-precision \
// RUN: -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-EXCESS

float f(float x, float y) {
  // CHECK: define{{.*}} float @f
  // CHECK: fadd float
  return 2.0f + x + y;
}

int getEvalMethod(void) {
  // CHECK-SRC: ret i32 0
  // CHECK-EXCESS: ret i32 2
  return __FLT_EVAL_METHOD__;
}
