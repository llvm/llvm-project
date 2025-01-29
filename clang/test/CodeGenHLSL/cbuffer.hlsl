// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute \
// RUN:   -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %struct.__cblayout_CBScalars = type <{ float, double, half, i64, i32, i16, i32, i64 }>
// CHECK: %struct.__cblayout_CBVectors = type <{ <3 x float>, <3 x double>, <2 x half>, <3 x i64>, <4 x i32>, <3 x i16>, <3 x i64> }>
// CHECK: %struct.__cblayout_CBArrays = type <{ [3 x float], [2 x <3 x double>], [2 x [2 x half]], [3 x i64], [2 x [3 x [4 x <4 x i32>]]], [1 x i16], [2 x i64], [4 x i32] }>
// CHECK: %struct.__cblayout_CBStructs = type { %struct.A, %struct.B, %struct.C, [5 x %struct.A], %struct.__cblayout_D, half, %struct.B, <3 x i16> }
// CHECK: %struct.A = type { <2 x float> }
// CHECK: %struct.C = type { i32, %struct.A }
// CHECK: %struct.__cblayout_D = type { [2 x [3 x %struct.B]] }
// CHECK: %struct.B = type { %struct.A, <3 x i16> }

cbuffer CBScalars : register(b1, space5) {
  float a1;
  double a2;
  float16_t a3;
  uint64_t a4;
  int a5;
  uint16_t a6;
  bool a7;
  int64_t a8;
}

// CHECK: @CBScalars.cb = external constant target("dx.CBuffer", %struct.__cblayout_CBScalars)
// CHECK: @a1 = external addrspace(2) global float, align 4
// CHECK: @a2 = external addrspace(2) global double, align 8
// CHECK: @a3 = external addrspace(2) global half, align 2
// CHECK: @a4 = external addrspace(2) global i64, align 8
// CHECK: @a5 = external addrspace(2) global i32, align 4
// CHECK: @a6 = external addrspace(2) global i16, align 2
// CHECK: @a7 = external addrspace(2) global i32, align 4
// CHECK: @a8 = external addrspace(2) global i64, align 8

cbuffer CBVectors {
  float3 b1;
  double3 b2;
  float16_t2 b3;
  uint64_t3 b4;
  int4 b5;
  uint16_t3 b6;
  int64_t3 b7;
  // FIXME: add s bool vectors after llvm-project/llvm#91639 is added
}

// CHECK: @CBVectors.cb = external constant target("dx.CBuffer", %struct.__cblayout_CBVectors)
// CHECK: @b1 = external addrspace(2) global <3 x float>, align 16
// CHECK: @b2 = external addrspace(2) global <3 x double>, align 32
// CHECK: @b3 = external addrspace(2) global <2 x half>, align 4
// CHECK: @b4 = external addrspace(2) global <3 x i64>, align 32
// CHECK: @b5 = external addrspace(2) global <4 x i32>, align 16
// CHECK: @b6 = external addrspace(2) global <3 x i16>, align 8
// CHECK: @b7 = external addrspace(2) global <3 x i64>, align 32

cbuffer CBArrays : register(b2) {
  float c1[3];
  double3 c2[2];
  float16_t c3[2][2];
  uint64_t c4[3];
  int4 c5[2][3][4];
  uint16_t c6[1];
  int64_t c7[2];
  bool c8[4];
}

// CHECK: @CBArrays.cb = external constant target("dx.CBuffer", %struct.__cblayout_CBArrays)
// CHECK: @c1 = external addrspace(2) global [3 x float], align 4
// CHECK: @c2 = external addrspace(2) global [2 x <3 x double>], align 32
// CHECK: @c3 = external addrspace(2) global [2 x [2 x half]], align 2
// CHECK: @c4 = external addrspace(2) global [3 x i64], align 8
// CHECK: @c5 = external addrspace(2) global [2 x [3 x [4 x <4 x i32>]]], align 16
// CHECK: @c6 = external addrspace(2) global [1 x i16], align 2
// CHECK: @c7 = external addrspace(2) global [2 x i64], align 8
// CHECK: @c8 = external addrspace(2) global [4 x i32], align 4

struct Empty {};

struct A {
  float2 f1;
};

struct B : A {
  uint16_t3 f2;
};

struct C {
  int i;
  A f3;
};

struct D {
  B array_of_B[2][3];
  Empty es;
};

// CHECK: @CBStructs.cb = external constant target("dx.CBuffer", %struct.__cblayout_CBStructs)
// CHECK: @a = external addrspace(2) global %struct.A, align 8
// CHECK: @b = external addrspace(2) global %struct.B, align 8
// CHECK: @c = external addrspace(2) global %struct.C, align 8
// CHECK: @array_of_A = external addrspace(2) global [5 x %struct.A], align 8
// CHECK: @d = external addrspace(2) global %struct.__cblayout_D, align 8
// CHECK: @e = external addrspace(2) global half, align 2

cbuffer CBStructs {
  A a;
  B b;
  C c;
  A array_of_A[5];
  D d;
  half e;
  B f;
  uint16_t3 g;
};

struct Test {
    float a, b;
};

// CHECK: @CBMix.cb = external constant target("dx.CBuffer", %struct.__cblayout_CBMix)
// CHECK: @test = external addrspace(2) global [2 x %struct.Test], align 4
// CHECK: @f1 = external addrspace(2) global float, align 4
// CHECK: @f2 = external addrspace(2) global [3 x [2 x <2 x float>]], align 8
// CHECK: @f3 = external addrspace(2) global float, align 4
// CHECK: @s = external addrspace(2) global %struct.anon, align 4
// CHECK: @dd = external addrspace(2) global double, align 8
// CHECK: @f4 = external addrspace(2) global float, align 4
// CHECK: @dv = external addrspace(2) global <1 x double>, align 8
// CHECK: @uv = external addrspace(2) global i16, align 2

cbuffer CBMix {
    Test test[2];
    float f1;
    float2 f2[3][2];
    float f3;
    struct { float c; } s;
    double dd;
    float f4;
    vector<double,1> dv;
    uint16_t uv;
};  

// CHECK: efine internal void @_init_resource_CBScalars.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[HANDLE1:.*]] = call target("dx.CBuffer", %struct.__cblayout_CBScalars)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s_struct.__cblayout_CBScalarsst(i32 5, i32 1, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", %struct.__cblayout_CBScalars) %[[HANDLE1]], ptr @CBScalars.cb, align 4

// CHECK: define internal void @_init_resource_CBArrays.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[HANDLE2:.*]] = call target("dx.CBuffer", %struct.__cblayout_CBArrays)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s_struct.__cblayout_CBArraysst(i32 0, i32 2, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", %struct.__cblayout_CBArrays) %[[HANDLE2]], ptr @CBArrays.cb, align 4

RWBuffer<float> Buf;

[numthreads(4,1,1)]
void main() {
  //Buf[0] = a1 + b1.z + c1[2] + a.f1.y;
  Buf[0] = a.f1.y;
}

// CHECK: define internal void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_resource_CBScalars.cb()
// CHECK-NEXT: call void @_init_resource_CBArrays.cb()

// CHECK: !hlsl.cbs = !{![[CBSCALARS:[0-9]+]], ![[CBVECTORS:[0-9]+]], ![[CBARRAYS:[0-9]+]], ![[CBSTRUCTS:[0-9]+]], ![[CBMIX:[0-9]+]]}
// CHECK: !hlsl.cblayouts = !{![[CBSCALARS_LAYOUT:[0-9]+]], ![[CBVECTORS_LAYOUT:[0-9]+]], ![[CBARRAYS_LAYOUT:[0-9]+]], ![[A_LAYOUT:[0-9]+]],
// CHECK-SAME: ![[B_LAYOUT:[0-9]+]], ![[C_LAYOUT:[0-9]+]], ![[D_LAYOUT:[0-9]+]], ![[CBSTRUCTS_LAYOUT:[0-9]+]], ![[TEST_LAYOUT:[0-9]+]],
// CHECK-SAME: ![[ANON_LAYOUT:[0-9]+]], ![[CBMIX_LAYOUT:[0-9]+]]}

// CHECK: ![[CBSCALARS]] = !{ptr @CBScalars.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4,
// CHECK-SAME: ptr addrspace(2) @a5, ptr addrspace(2) @a6, ptr addrspace(2) @a7, ptr addrspace(2) @a8}

// CHECK: ![[CBVECTORS]] = !{ptr @CBVectors.cb, ptr addrspace(2) @b1, ptr addrspace(2) @b2, ptr addrspace(2) @b3, ptr addrspace(2) @b4,
// CHECK-SAME: ptr addrspace(2) @b5, ptr addrspace(2) @b6, ptr addrspace(2) @b7}

// CHECK: ![[CBARRAYS]] = !{ptr @CBArrays.cb, ptr addrspace(2) @c1, ptr addrspace(2) @c2, ptr addrspace(2) @c3, ptr addrspace(2) @c4, 
// CHECK-SAME: ptr addrspace(2) @c5, ptr addrspace(2) @c6, ptr addrspace(2) @c7, ptr addrspace(2) @c8}

// CHECK: ![[CBSTRUCTS]] = !{ptr @CBStructs.cb, ptr addrspace(2) @a, ptr addrspace(2) @b, ptr addrspace(2) @c, ptr addrspace(2) @array_of_A, 
// CHECK-SAME: ptr addrspace(2) @d, ptr addrspace(2) @e, ptr addrspace(2) @f, ptr addrspace(2) @g}

// CHECK: ![[CBMIX]] = !{ptr @CBMix.cb, ptr addrspace(2) @test, ptr addrspace(2) @f1, ptr addrspace(2) @f2, ptr addrspace(2) @f3,
// CHECK-SAME: ptr addrspace(2) @s, ptr addrspace(2) @dd, ptr addrspace(2) @f4, ptr addrspace(2) @dv, ptr addrspace(2) @uv}

// CHECK: ![[CBSCALARS_LAYOUT]] = !{!"struct.__cblayout_CBScalars", i32 56, i32 0, i32 8, i32 16, i32 24, i32 32, i32 36, i32 40, i32 48}
// CHECK: ![[CBVECTORS_LAYOUT]] = !{!"struct.__cblayout_CBVectors", i32 136, i32 0, i32 16, i32 40, i32 48, i32 80, i32 96, i32 112}
                                  
// CHECK: ![[CBARRAYS_LAYOUT]] = !{!"struct.__cblayout_CBArrays", i32 708, i32 0, i32 48, i32 112, i32 176, i32 224, i32 608, i32 624, i32 656}

// CHECK: ![[A_LAYOUT]] = !{!"struct.A", i32 8, i32 0}
// CHECK: ![[B_LAYOUT]] = !{!"struct.B", i32 14, i32 0, i32 8}
// CHECK: ![[C_LAYOUT]] = !{!"struct.C", i32 24, i32 0, i32 16}
// CHECK: ![[D_LAYOUT]] = !{!"struct.__cblayout_D", i32 94, i32 0}
// CHECK: ![[CBSTRUCTS_LAYOUT]] = !{!"struct.__cblayout_CBStructs", i32 262, i32 0, i32 16, i32 32, i32 64, i32 144, i32 238, i32 240, i32 256}

// CHECK: ![[TEST_LAYOUT]] = !{!"struct.Test", i32 8, i32 0, i32 4}
// CHECK: ![[ANON_LAYOUT]] = !{!"struct.anon", i32 4, i32 0}
// CHECK: ![[CBMIX_LAYOUT]] = !{!"struct.__cblayout_CBMix", i32 162, i32 0, i32 24, i32 32, i32 120, i32 128, i32 136, i32 144, i32 152, i32 160}
