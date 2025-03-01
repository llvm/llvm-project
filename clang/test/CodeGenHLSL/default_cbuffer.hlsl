// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute \
// RUN:   -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %"__cblayout_$Globals" = type <{ float, float, target("dx.Layout", %__cblayout_S, 4, 0) }>
// CHECK: %__cblayout_S = type <{ float }>

// CHECK-DAG: @"$Globals.cb" = external constant target("dx.CBuffer", target("dx.Layout", %"__cblayout_$Globals", 20, 0, 4, 16))
// CHECK-DAG: @a = external addrspace(2) global float
// CHECK-DAG: @g = external addrspace(2) global float
// CHECK-DAG: @h = external addrspace(2) global target("dx.Layout", %__cblayout_S, 4, 0), align 4

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
// CHECK: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(2) @a, ptr addrspace(2) @g, ptr addrspace(2) @h}
