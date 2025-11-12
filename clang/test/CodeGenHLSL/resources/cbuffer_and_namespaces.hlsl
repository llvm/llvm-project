// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure cbuffer inside namespace works.

// CHECK: %"n0::n1::__cblayout_A" = type <{ float }>
// CHECK: %"n0::__cblayout_B" = type <{ float }>
// CHECK: %"n0::n2::__cblayout_C" = type <{ float, target("dx.Layout", %"n0::Foo", 4, 0) }>
// CHECK: %"n0::Foo" = type <{ float }>

// CHECK: @A.cb = global target("dx.CBuffer", target("dx.Layout", %"n0::n1::__cblayout_A", 4, 0))
// CHECK: @_ZN2n02n11aE = external hidden addrspace(2) global float, align 4

// CHECK: @B.cb = global target("dx.CBuffer", target("dx.Layout", %"n0::__cblayout_B", 4, 0))
// CHECK: @_ZN2n01aE = external hidden addrspace(2) global float, align 4

// CHECK: @C.cb = global target("dx.CBuffer", target("dx.Layout", %"n0::n2::__cblayout_C", 20, 0, 16))
// CHECK: @_ZN2n02n21aE = external hidden addrspace(2) global float, align 4
// CHECK: external hidden addrspace(2) global target("dx.Layout", %"n0::Foo", 4, 0), align 1

namespace n0 {
  struct Foo {
    float f;
  };

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
      Foo b;
    }
  }
}

float foo() {
  // CHECK: load float, ptr addrspace(2) @_ZN2n02n11aE, align 4
  // CHECK: load float, ptr addrspace(2) @_ZN2n01aE, align 4
  // CHECK: load float, ptr addrspace(2) @_ZN2n02n21aE, align 4
  return n0::n1::a + n0::a + n0::n2::a;
}

[numthreads(4,1,1)]
void main() {}

// CHECK: !hlsl.cbs = !{![[A:[0-9]+]], ![[B:[0-9]+]], ![[C:[0-9]+]]}
// CHECK: [[A]] = !{ptr @A.cb, ptr addrspace(2) @_ZN2n02n11aE}
// CHECK: [[B]] = !{ptr @B.cb, ptr addrspace(2) @_ZN2n01aE}
// CHECK: [[C]] = !{ptr @C.cb, ptr addrspace(2) @_ZN2n02n21aE, ptr addrspace(2) @_ZN2n02n21bE}
