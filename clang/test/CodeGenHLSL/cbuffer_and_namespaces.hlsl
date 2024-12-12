// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure cbuffer inside namespace works.
// CHECK: @A.cb = external constant target("dx.CBuffer", { float }, 4, 0)
// CHECK: @B.cb = external constant target("dx.CBuffer", { float }, 4, 0)
// CHECK: @C.cb = external constant target("dx.CBuffer", { float }, 4, 0)

namespace n0 {
  namespace n1 {
    cbuffer A {
      float a;
    }
  }
  cbuffer B {
    float a;
  }
  namespace n2 {
    cbuffer C {
      float a;
    }
  }
}

float foo() {
// CHECK: %[[HANDLE1:[0-9]+]] = load target("dx.CBuffer", { float }, 4, 0), ptr @A.cb, align 4
// CHECK: %[[PTR1:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_f32s_4_0t(target("dx.CBuffer", { float }, 4, 0) %[[HANDLE1]], i32 0)
// CHECK: %[[VAL1:[0-9]+]] = load float, ptr %[[PTR1]], align 4

// CHECK: %[[HANDLE2:[0-9]+]] = load target("dx.CBuffer", { float }, 4, 0), ptr @B.cb, align 4
// CHECK: %[[PTR2:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_f32s_4_0t(target("dx.CBuffer", { float }, 4, 0) %[[HANDLE2]], i32 0)
// CHECK: %[[VAL2:[0-9]+]] = load float, ptr %[[PTR2]], align 4

// CHECK: %add = fadd float %[[VAL1]], %[[VAL2]]

// CHECK: %[[HANDLE3:[0-9]+]] = load target("dx.CBuffer", { float }, 4, 0), ptr @C.cb, align 4
// CHECK: %[[PTR3:[0-9]+]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.CBuffer_sl_f32s_4_0t(target("dx.CBuffer", { float }, 4, 0) %[[HANDLE3]], i32 0)
// CHECK: %[[VAL3:[0-9]+]] = load float, ptr %[[PTR3]], align 4

// CHECK: %add1 = fadd float %add, %[[VAL3]]

  return n0::n1::a + n0::a + n0::n2::a;
}

[numthreads(4,1,1)]
void main() {}
