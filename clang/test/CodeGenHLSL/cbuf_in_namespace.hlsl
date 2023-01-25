// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure cbuffer inside namespace works.
// CHECK: @[[CB:.+]] = external constant { float }
// CHECK: @[[TB:.+]] = external constant { float }
namespace n0 {
namespace n1 {
  cbuffer A {
    float a;
  }
}
  tbuffer B {
    float b;
  }
}

float foo() {
// CHECK: load float, ptr @[[CB]], align 4
// CHECK: load float, ptr @[[TB]], align 4
  return n0::n1::a + n0::b;
}
