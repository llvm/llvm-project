// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

typedef union {
  int    i;
  float  f;
} MyUnion;
void unionf(MyUnion a);
void uniontest(float a) {
  f((MyUnion)1.0f);
// CHECK: store float 1.000000e+00
}

