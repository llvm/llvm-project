// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure cbuffer inside namespace works.

// CHECK: %"class.n0::n1::A" = type { float }
// CHECK: %"class.n0::B" = type { float }
// CHECK: %"class.n0::n2::C" = type { float }

// CHECK: @A.cb = external constant target("dx.CBuffer", %"class.n0::n1::A", 4, 0)
// CHECK: @B.cb = external constant target("dx.CBuffer", %"class.n0::B", 4, 0)
// CHECK: @C.cb = external constant target("dx.CBuffer", %"class.n0::n2::C", 4, 0)

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
// CHECK: %[[HANDLE1:[0-9]+]] = load target("dx.CBuffer", %"class.n0::n1::A", 4, 0), ptr @A.cb, align 4
// CHECK: %[[PTR1:[0-9]+]] = call ptr @"llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.n0::n1::As_4_0t"(target("dx.CBuffer", %"class.n0::n1::A", 4, 0) %[[HANDLE1]], i32 0)
// CHECK: %_ZN2n02n11aE = getelementptr %"class.n0::n1::A", ptr %[[PTR1]], i32 0, i32 0
// CHECK: %[[VAL1:[0-9]+]] = load float, ptr %_ZN2n02n11aE, align 4

// CHECK: %[[HANDLE2:[0-9]+]] = load target("dx.CBuffer", %"class.n0::B", 4, 0), ptr @B.cb, align 4
// CHECK: %[[PTR2:[0-9]+]] = call ptr @"llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.n0::Bs_4_0t"(target("dx.CBuffer", %"class.n0::B", 4, 0) %[[HANDLE2]], i32 0)
// CHECK: %_ZN2n01aE = getelementptr %"class.n0::B", ptr %[[PTR2]], i32 0, i32 0
// CHECK: %[[VAL2:[0-9]+]] = load float, ptr %_ZN2n01aE, align 4

// CHECK: %add = fadd float %[[VAL1]], %[[VAL2]]

// CHECK: %[[HANDLE3:[0-9]+]] = load target("dx.CBuffer", %"class.n0::n2::C", 4, 0), ptr @C.cb, align 4
// CHECK: %[[PTR3:[0-9]+]] = call ptr @"llvm.dx.resource.getpointer.p0.tdx.CBuffer_s_class.n0::n2::Cs_4_0t"(target("dx.CBuffer", %"class.n0::n2::C", 4, 0) %[[HANDLE3]], i32 0)
// CHECK: %_ZN2n02n21aE = getelementptr %"class.n0::n2::C", ptr %[[PTR3]], i32 0, i32 0
// CHECK: %[[VAL3:[0-9]+]] = load float, ptr %_ZN2n02n21aE, align 4

// CHECK: %add1 = fadd float %add, %[[VAL3]]

  return n0::n1::a + n0::a + n0::n2::a;
}

[numthreads(4,1,1)]
void main() {}
