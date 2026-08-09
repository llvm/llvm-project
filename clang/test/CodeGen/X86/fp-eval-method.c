// RUN: %clang_cc1 -triple i386-unknown-netbsd -emit-llvm -o - %s \
// RUN: | FileCheck %s

// RUN: %clang_cc1 -triple i386--linux -emit-llvm -o - %s \
// RUN: | FileCheck %s

float f(float x, float y) {
  // CHECK: define{{.*}} float @f
  // CHECK: fadd float
  return 2.0f + x + y;
}

int getEvalMethod(void) {
  // CHECK: ret i32 0
  return __FLT_EVAL_METHOD__;
}
