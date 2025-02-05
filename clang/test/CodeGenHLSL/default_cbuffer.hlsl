// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute \
// RUN:   -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %"struct.__cblayout_$Globals" = type { float, float, %struct.__cblayout_S }
// CHECK: %struct.__cblayout_S = type { float }

// CHECK-DAG: @"$Globals.cb" = external constant target("dx.CBuffer", %"struct.__cblayout_$Globals")
// CHECK-DAG: @a = external addrspace(2) global float
// CHECK-DAG: @g = external addrspace(2) global float
// CHECK-DAG: @h = external addrspace(2) global %struct.__cblayout_S

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

// CHECK: !hlsl.cblayouts = !{![[S_LAYOUT:.*]], ![[CB_LAYOUT:.*]]}
// CHECK: !hlsl.cbs = !{![[CB:.*]]}

// CHECK: ![[S_LAYOUT]] = !{!"struct.__cblayout_S", i32 4, i32 0}
// CHECK: ![[CB_LAYOUT]] = !{!"struct.__cblayout_$Globals", i32 20, i32 0, i32 4, i32 16}
// CHECK: ![[CB]] = !{ptr @"$Globals.cb", ptr addrspace(2) @a, ptr addrspace(2) @g, ptr addrspace(2) @h}