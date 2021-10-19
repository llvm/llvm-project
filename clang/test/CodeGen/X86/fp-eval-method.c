// RUN: %clang_cc1 -triple i386-unknown-netbsd6 -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK

// RUN: %clang_cc1 -triple i386-unknown-netbsd7 -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK-EXT

// RUN: %clang_cc1 -triple i386--linux -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK-EXT

float f(float x, float y) {
  // CHECK: define{{.*}} float @f
  // CHECK: fadd float
  return 2.0f + x + y;
}

int getEvalMethod() {
  // CHECK: ret i32 1
  // CHECK-EXT: ret i32 2
  return __FLT_EVAL_METHOD__;
}
