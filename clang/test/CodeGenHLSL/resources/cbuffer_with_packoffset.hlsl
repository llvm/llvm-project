// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK:      %__cblayout_CB = type <{
// CHECK-SAME:   target("dx.Padding", 16),
// CHECK-SAME:   float,
// CHECK-SAME:   target("dx.Padding", 68),
// CHECK-SAME:   <2 x i32>,
// CHECK-SAME    target("dx.Padding", 72),
// CHECK-SAME:   double
// CHECK-SAME: }>
// CHECK:      %__cblayout_CB_1 = type <{
// CHECK-SAME:   target("dx.Padding", 80),
// CHECK-SAME:   <2 x float>,
// CHECK-SAME:   float
// CHECK-SAME: }>

// CHECK-DAG: @CB.cb = global target("dx.CBuffer", %__cblayout_CB)
// CHECK-DAG: @a = external hidden addrspace(2) global float, align 4
// CHECK-DAG: @b = external hidden addrspace(2) global double, align 8
// CHECK-DAG: @c = external hidden addrspace(2) global <2 x i32>, align 8
// CHECK: @CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1

cbuffer CB : register(b1, space3) {
  float a : packoffset(c1.x);
  double b : packoffset(c10.z);
  int2 c : packoffset(c5.z);
}

// CHECK-DAG: @CB.cb.1 = global target("dx.CBuffer", %__cblayout_CB_1)
// CHECK-DAG: @x = external hidden addrspace(2) global float, align 4
// CHECK-DAG: @y = external hidden addrspace(2) global <2 x float>, align 8

// Missing packoffset annotation will produce a warning.
// Element x will be placed after the element y that has an explicit packoffset.
cbuffer CB : register(b0) {
  float x;
  float2 y : packoffset(c5);
}

// CHECK: define internal void @_init_buffer_CB.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB.cb_h = call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_CBst(i32 3, i32 1, i32 1, i32 0, ptr @CB.str)

float foo() {
  // CHECK: load float, ptr addrspace(2) @a, align 4
  // CHECK: load double, ptr addrspace(2) @b, align 8
  return a + b;
}
// CHECK: define internal void @_GLOBAL__sub_I_cbuffer_with_packoffset.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_buffer_CB.cb()

[numthreads(4,1,1)]
void main() {
  foo();
}

// CHECK: !hlsl.cbs = !{![[CB1:[0-9]+]], ![[CB2:[0-9]+]]}
// CHECK: ![[CB1]] = !{ptr @CB.cb, ptr addrspace(2) @a, ptr addrspace(2) @c, ptr addrspace(2) @b}
// CHECK: ![[CB2]] = !{ptr @CB.cb.1, ptr addrspace(2) @y, ptr addrspace(2) @x}
