// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %"__cblayout_$Globals" = type <{ i32, float, [4 x double], <4 x i32>, <4 x float>,
// CHECK-SAME: target("dx.Layout", %S, 8, 0) }>
// CHECK: %S = type <{ <2 x float> }>

// CHECK-DAG: @b = external hidden addrspace(2) global float, align 4
// CHECK-DAG: @d = external hidden addrspace(2) global <4 x i32>, align 16
// CHECK-DAG: @"$Globals.cb" = global target("dx.CBuffer",
// CHECK-DAG-SAME: target("dx.Layout", %"__cblayout_$Globals", 144, 120, 16, 32, 64, 128, 112))
// CHECK-DAG: @a = external hidden addrspace(2) global i32, align 4
// CHECK-DAG: @c = external hidden addrspace(2) global [4 x double], align 8
// CHECK-DAG: @e = external hidden addrspace(2) global <4 x float>, align 16
// CHECK-DAG: @s = external hidden addrspace(2) global target("dx.Layout", %S, 8, 0), align 1

struct S {
  float2 v;
};

int a;
float b : register(c1);
double c[4] : register(c2);
int4 d : register(c4);
float4 e;
S s : register(c7);

RWBuffer<float> Buf;

[numthreads(4,1,1)]
void main() {
  Buf[0] = a;
}

// CHECK: !hlsl.cbs = !{![[CB:.*]]}
// CHECK: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(2) @a, ptr addrspace(2) @b, ptr addrspace(2) @c,
// CHECK-SAME: ptr addrspace(2) @d, ptr addrspace(2) @e, ptr addrspace(2) @s}
