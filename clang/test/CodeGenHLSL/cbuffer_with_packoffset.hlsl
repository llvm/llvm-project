// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %__cblayout_CB = type <{ float, double, <2 x i32> }>

// CHECK: @CB.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 176, 16, 168, 88))
// CHECK: @a = external addrspace(2) global float, align 4
// CHECK: @b = external addrspace(2) global double, align 8
// CHECK: @c = external addrspace(2) global <2 x i32>, align 8

cbuffer CB : register(b1, space3) {
  float a : packoffset(c1.x);
  double b : packoffset(c10.z);
  int2 c : packoffset(c5.z);
}

// CHECK: define internal void @_init_resource_CB.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 176, 16, 168, 88))
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBs_176_16_168_88tt(i32 3, i32 1, i32 1, i32 0, i1 false)

float foo() {
  // CHECK: load float, ptr addrspace(2) @a, align 4
  // CHECK: load double, ptr addrspace(2) @b, align 8
  return a + b;
}
// CHECK: define internal void @_GLOBAL__sub_I_cbuffer_with_packoffset.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_resource_CB.cb()

[numthreads(4,1,1)]
void main() {
  foo();
}

// CHECK: !hlsl.cbs = !{![[CB:[0-9]+]]}
// CHECK: ![[CB]] = !{ptr @CB.cb, ptr addrspace(2) @a, ptr addrspace(2) @b, ptr addrspace(2) @c}
