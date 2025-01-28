// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %struct.__cblayout_A = type { float }

// CHECK: @A.cb = external constant target("dx.CBuffer", %struct.__cblayout_A)
// CHECK: @a = external addrspace(2) global float, align 4
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

extern float bar() {
  return foo();
}

// CHECK: !hlsl.cbs = !{![[CB:[0-9]+]]}
// CHECK: !hlsl.cblayouts = !{![[CB_LAYOUT:[0-9]+]]}

// CHECK: ![[CB]] = !{ptr @A.cb, ptr addrspace(2) @a}
// CHECK: ![[CB_LAYOUT]] = !{!"struct.__cblayout_A", i32 4, i32 0}
