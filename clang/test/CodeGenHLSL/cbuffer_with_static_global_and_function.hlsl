// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %__cblayout_A = type <{ float }>

// CHECK-NOT: @B.cb
cbuffer B {
  // intentionally empty
}

// CHECK-DAG: @_ZL1b = internal addrspace(10) global float 3.000000e+00, align 4
static float b = 3;

// CHECK-DAG: @A.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_A, 4, 0))
cbuffer A {
// CHECK-DAG: @a = external addrspace(2) global float, align 4
  float a;

// CHECK: define {{.*}} float @_Z3foov() #0 {
// CHECK-DAG: load float, ptr addrspace(2) @a, align 4
// CHECK-DAG: load float, ptr addrspacecast (ptr addrspace(10) @_ZL1b to ptr), align 4
  float foo() { return a + b; }
}


extern float bar() {
  return foo();
}

// CHECK: !hlsl.cbs = !{![[CB:[0-9]+]]}
// CHECK: ![[CB]] = !{ptr @A.cb, ptr addrspace(2) @a}
