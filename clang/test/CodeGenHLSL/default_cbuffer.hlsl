// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

// DXIL: %"__cblayout_$Globals" = type <{ float, float, target("dx.Layout", %__cblayout_S, 4, 0) }>
// SPIRV: %"__cblayout_$Globals" = type <{ float, float, target("spirv.Layout", %__cblayout_S, 4, 0) }>
// CHECK: %__cblayout_S = type <{ float }>

// DXIL-DAG: @"$Globals.cb" = global target("dx.CBuffer", target("dx.Layout", %"__cblayout_$Globals", 20, 0, 4, 16))
// DXIL-DAG: @a = external hidden addrspace(2) global float
// DXIL-DAG: @g = external hidden addrspace(2) global float
// DXIL-DAG: @h = external hidden addrspace(2) global target("dx.Layout", %__cblayout_S, 4, 0), align 4

// SPIRV-DAG: @"$Globals.cb" = global target("spirv.VulkanBuffer", target("spirv.Layout", %"__cblayout_$Globals", 20, 0, 4, 16), 2, 0)
// SPIRV-DAG: @a = external hidden addrspace(12) global float
// SPIRV-DAG: @g = external hidden addrspace(12) global float
// SPIRV-DAG: @h = external hidden addrspace(12) global target("spirv.Layout", %__cblayout_S, 4, 0), align 8

struct EmptyStruct {
};

struct S {
  RWBuffer<float> buf;
  EmptyStruct es;
  float ea[0];
  float b;
};

float a;
RWBuffer<float> b;
EmptyStruct c;
float d[0];
RWBuffer<float> e[2];
groupshared float f;
float g;
S h;

RWBuffer<float> Buf;

[numthreads(4,1,1)]
void main() {
  Buf[0] = a;
}

// CHECK: !hlsl.cbs = !{![[CB:.*]]}
// DXIL: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(2) @a, ptr addrspace(2) @g, ptr addrspace(2) @h}
// SPIRV: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(12) @a, ptr addrspace(12) @g, ptr addrspace(12) @h}
