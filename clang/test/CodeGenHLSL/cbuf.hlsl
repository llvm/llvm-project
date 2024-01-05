// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: @[[CB:.+]] = external constant { float, double }
cbuffer A : register(b0, space2) {
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

// CHECK: !hlsl.cbufs = !{![[CBMD:[0-9]+]]}
// CHECK: !hlsl.srvs = !{![[TBMD:[0-9]+]]}
// CHECK: ![[CBMD]] = !{ptr @[[CB]], i32 13, i32 0, i1 false, i32 0, i32 2}
// CHECK: ![[TBMD]] = !{ptr @[[TB]], i32 15, i32 0, i1 false, i32 2, i32 1}
