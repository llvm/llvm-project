// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK-DAG: @[[CB:.+]] = external constant { float }

cbuffer A {
    float a;
  // CHECK-DAG:@b = internal global float 3.000000e+00, align 4
  static float b = 3;
  // CHECK:load float, ptr @[[CB]], align 4
  // CHECK:load float, ptr @b, align 4
  float foo() { return a + b; }
}

float bar() {
  return foo();
}
