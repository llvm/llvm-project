// RUN: %clang_cc1 -fexperimental-strict-floating-point  \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - %s  \
// RUN: | FileCheck %s -check-prefixes=CHECK

// RUN: %clang_cc1 -triple i386--linux -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK-EXT

// RUN: %clang_cc1 -fexperimental-strict-floating-point  \
// RUN: -mreassociate -freciprocal-math -ffp-contract=fast \
// RUN: -ffast-math -triple x86_64-linux-gnu \
// RUN: -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK-FAST

// RUN: %clang_cc1 -triple i386--linux -mreassociate -freciprocal-math \
// RUN: -ffp-contract=fast -ffast-math -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK-FAST

float a = 1.0f, b = 2.0f, c = 3.0f;
#pragma float_control(precise, off)
float res2 = a + b + c;
int val3 = __FLT_EVAL_METHOD__;
#pragma float_control(precise, on)
float res3 = a + b + c;
int val4 = __FLT_EVAL_METHOD__;

// CHECK: @val3 = global i32 -1
// CHECK: @val4 = global i32 0

// CHECK-EXT: @val3 = global i32 -1
// CHECK-EXT: @val4 = global i32 2

// CHECK-FAST: @val3 = global i32 -1
// CHECK-FAST: @val4 = global i32 -1

float res;
int add(float a, float b, float c) {
  // CHECK: fadd float
  // CHECK: load float, float*
  // CHECK: fadd float
  // CHECK: store float
  // CHECK: ret i32 0
  res = a + b + c;
  return __FLT_EVAL_METHOD__;
}

int add_precise(float a, float b, float c) {
#pragma float_control(precise, on)
  // CHECK: fadd float
  // CHECK: load float, float*
  // CHECK: fadd float
  // CHECK: store float
  // CHECK: ret i32 0
  res = a + b + c;
  return __FLT_EVAL_METHOD__;
}

#pragma float_control(push)
#pragma float_control(precise, on)
int add_precise_1(float a, float b, float c) {
  // CHECK: fadd float
  // CHECK: load float, float*
  // CHECK: fadd float
  // CHECK: store float
  // CHECK: ret i32 0
  res = a + b + c;
  return __FLT_EVAL_METHOD__;
}
#pragma float_control(pop)

int add_not_precise(float a, float b, float c) {
  // Fast-math is enabled with this pragma.
#pragma float_control(precise, off)
  // CHECK: fadd fast float
  // CHECK: load float, float*
  // CHECK: fadd fast float
  // CHECK: float {{.*}}, float*
  // CHECK: ret i32 -1
  res = a + b + c;
  return __FLT_EVAL_METHOD__;
}

#pragma float_control(push)
// Fast-math is enabled with this pragma.
#pragma float_control(precise, off)
int add_not_precise_1(float a, float b, float c) {
  // CHECK: fadd fast float
  // CHECK: load float, float*
  // CHECK: fadd fast float
  // CHECK: float {{.*}}, float*
  // CHECK: ret i32 -1
  res = a + b + c;
  return __FLT_EVAL_METHOD__;
}
#pragma float_control(pop)

int getFPEvalMethod() {
  // CHECK: ret i32 0
  return __FLT_EVAL_METHOD__;
}

float res1;
int whatever(float a, float b, float c) {
#pragma float_control(precise, off)
  // CHECK: load float, float*
  // CHECK: fadd fast float
  // CHECK: store float {{.*}}, float*
  // CHECK: store i32 -1
  // CHECK: store i32 0
  // CHECK: ret i32 -1
  res1 = a + b + c;
  int val1 = __FLT_EVAL_METHOD__;
  {
#pragma float_control(precise, on)
    int val2 = __FLT_EVAL_METHOD__;
  }
  return __FLT_EVAL_METHOD__;
}
