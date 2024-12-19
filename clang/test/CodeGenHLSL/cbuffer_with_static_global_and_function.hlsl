// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %class.A = type { float }

// CHECK-DAG: @_ZL1b = internal global float 3.000000e+00, align 4
// CHECK-DAG: @A.cb = external constant target("dx.CBuffer", %class.A, 4, 0)
// CHECK-NOT: @B.cb

cbuffer A {
  float a;
  static float b = 3;
  float foo() { return a + b; }
}

cbuffer B {
  // intentionally empty
}


// CHECK: define noundef float @_Z3foov() #0 {
// CHECK: %[[HANDLE:[0-9]+]] = load target("dx.CBuffer", %class.A, 4, 0), ptr @A.cb, align 4
// CHECK: %[[PTR:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.As_4_0t(target("dx.CBuffer", %class.A, 4, 0) %[[HANDLE]], i32 0)
// CHECK: %a = getelementptr %class.A, ptr %[[PTR]], i32 0, i32 0
// CHECK: %[[VAL1:[0-9]+]] = load float, ptr %a, align 4
// CHECK: %[[VAL2:[0-9]+]] = load float, ptr @_ZL1b, align 4
// CHECK: %add = fadd float %[[VAL1]], %[[VAL2]]

extern float bar() {
  return foo();
}
