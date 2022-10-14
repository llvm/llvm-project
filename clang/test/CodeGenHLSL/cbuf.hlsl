// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: @[[CB:.+]] = external constant { float, double }
cbuffer A : register(b0, space1) {
  float a;
  double b;
}

// CHECK: @[[TB:.+]] = external constant { float, double }
tbuffer A : register(t2, space1) {
  float c;
  double d;
}

float foo() {
// CHECK: load float, ptr @[[CB]], align 4
// CHECK: load double, ptr getelementptr inbounds ({ float, double }, ptr @[[CB]], i32 0, i32 1), align 8
// CHECK: load float, ptr @[[TB]], align 4
// CHECK: load double, ptr getelementptr inbounds ({ float, double }, ptr @[[TB]], i32 0, i32 1), align 8
  return a + b + c*d;
}
