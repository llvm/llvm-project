// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure cbuffer inside namespace works.

// CHECK: %"struct.n0::n1::__cblayout_A" = type { float }
// CHECK: %"struct.n0::__cblayout_B" = type { float }
// CHECK: %"struct.n0::n2::__cblayout_C" = type { float, %"struct.n0::Foo" }
// CHECK: %"struct.n0::Foo" = type { float }

// CHECK: @A.cb = external constant target("dx.CBuffer", %"struct.n0::n1::__cblayout_A")
// CHECK: @_ZN2n02n11aE = external addrspace(2) global float, align 4

// CHECK: @B.cb = external constant target("dx.CBuffer", %"struct.n0::__cblayout_B")
// CHECK: @_ZN2n01aE = external addrspace(2) global float, align 4

// CHECK: @C.cb = external constant target("dx.CBuffer", %"struct.n0::n2::__cblayout_C")
// CHECK: @_ZN2n02n21aE = external addrspace(2) global float, align 4
// CHECK: @_ZN2n02n21bE = external addrspace(2) global %"struct.n0::Foo", align 4

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
// CHECK: !hlsl.cblayouts = !{![[A_LAYOUT:[0-9]+]], ![[B_LAYOUT:[0-9]+]], ![[FOO_LAYOUT:[0-9]+]], ![[C_LAYOUT:[0-9]+]]}

// CHECK: [[A]] = !{ptr @A.cb, ptr addrspace(2) @_ZN2n02n11aE}
// CHECK: [[B]] = !{ptr @B.cb, ptr addrspace(2) @_ZN2n01aE}
// CHECK: [[C]] = !{ptr @C.cb, ptr addrspace(2) @_ZN2n02n21aE, ptr addrspace(2) @_ZN2n02n21bE}

// CHECK: ![[A_LAYOUT]] = !{!"struct.n0::n1::__cblayout_A", i32 4, i32 0}
// CHECK: ![[B_LAYOUT]] = !{!"struct.n0::__cblayout_B", i32 4, i32 0}
// CHECK: ![[FOO_LAYOUT]] = !{!"struct.n0::Foo", i32 4, i32 0}
// CHECK: ![[C_LAYOUT]] = !{!"struct.n0::n2::__cblayout_C", i32 20, i32 0, i32 16}
