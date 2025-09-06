// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %__cblayout_A = type <{ float }>

// CHECK: @A.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_A, 4, 0))
// CHECK: @a = external hidden addrspace(2) global float, align 4
// CHECK-DAG: @_ZL1b = internal global float 3.000000e+00, align 4
// CHECK-NOT: @B.cb

cbuffer A {
  float a;
  static float b = 3;
  float foo() { return a + b; }
}

cbuffer B {
  // intentionally empty
}

// CHECK: define {{.*}} float @_Z3foov() #0 {
// CHECK: load float, ptr addrspace(2) @a, align 4
// CHECK: load float, ptr @_ZL1b, align 4

extern float bar() {
  return foo();
}

// CHECK: !hlsl.cbs = !{![[CB:[0-9]+]]}
// CHECK: ![[CB]] = !{ptr @A.cb, ptr addrspace(2) @a}
