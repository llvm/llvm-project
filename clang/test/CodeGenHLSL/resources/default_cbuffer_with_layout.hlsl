// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK:      %"__cblayout_$Globals" = type <{
// CHECK-SAME:   float,
// CHECK-SAME:   target("dx.Padding", 12),
// CHECK-SAME:   <{ [3 x <{ double, target("dx.Padding", 8) }>], double }>,
// CHECK-SAME:   target("dx.Padding", 8),
// CHECK-SAME:   <4 x i32>,
// CHECK-SAME:   %S
// CHECK-SAME:   i32,
// CHECK-SAME:   target("dx.Padding", 4),
// CHECK-SAME:   <4 x float>
// CHECK-SAME: }>

// CHECK: %S = type <{ <2 x float> }>

// CHECK-DAG: @"$Globals.cb" = global target("dx.CBuffer", %"__cblayout_$Globals")
// CHECK-DAG: @a = external hidden addrspace(2) global i32, align 4
// CHECK-DAG: @b = external hidden addrspace(2) global float, align 4
// CHECK-DAG: @c = external hidden addrspace(2) global <{ [3 x <{ double, target("dx.Padding", 8) }>], double }>, align 8
// CHECK-DAG: @d = external hidden addrspace(2) global <4 x i32>, align 16
// CHECK-DAG: @e = external hidden addrspace(2) global <4 x float>, align 16
// CHECK-DAG: @s = external hidden addrspace(2) global %S, align 1

struct S {
  float2 v;
};

int a;
float b : register(c1);
int4 d : register(c6);
double c[4] : register(c2);
float4 e;
S s : register(c7);

RWBuffer<float> Buf;

[numthreads(4,1,1)]
void main() {
  Buf[0] = a;
}

// CHECK: !hlsl.cbs = !{![[CB:.*]]}
// CHECK: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(2) @b, ptr addrspace(2) @c, ptr addrspace(2) @d, ptr addrspace(2) @s, ptr addrspace(2) @a, ptr addrspace(2) @e}
