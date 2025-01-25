// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK-DAG: @[[CB:.+]] = external constant { float }

// CHECK-DAG:@_ZL1b = internal addrspace(10) global float 3.000000e+00, align 4
static float b = 3;

cbuffer A {
  float a;
  // CHECK:load float, ptr @[[CB]], align 4
  // CHECK:load float, ptr addrspacecast (ptr addrspace(10) @_ZL1b to ptr), align 4
  float foo() { return a + b; }
}

float bar() {
  return foo();
}
