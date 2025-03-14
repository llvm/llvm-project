// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute \
// RUN:   -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %__cblayout_CBScalars = type <{ float, double, half, i64, i32, i16, i32, i64 }>
// CHECK: %__cblayout_CBVectors = type <{ <3 x float>, <3 x double>, <2 x half>, <3 x i64>, <4 x i32>, <3 x i16>, <3 x i64> }>
// CHECK: %__cblayout_CBArrays = type <{ [3 x float], [2 x <3 x double>], [2 x [2 x half]], [3 x i64], [2 x [3 x [4 x <4 x i32>]]], [1 x i16], [2 x i64], [4 x i32] }>
// CHECK: %__cblayout_CBStructs = type <{ target("dx.Layout", %A, 8, 0), target("dx.Layout", %B, 14, 0, 8), 
// CHECK-SAME: target("dx.Layout", %C, 24, 0, 16), [5 x target("dx.Layout", %A, 8, 0)], 
// CHECK-SAME: target("dx.Layout", %__cblayout_D, 94, 0), half, <3 x i16> }>

// CHECK: %A = type <{ <2 x float> }>
// CHECK: %B = type <{ <2 x float>, <3 x i16> }>
// CHECK: %C = type <{ i32, target("dx.Layout", %A, 8, 0) }>
// CHECK: %__cblayout_D = type <{ [2 x [3 x target("dx.Layout", %B, 14, 0, 8)]] }>

// CHECK: %__cblayout_CBMix = type <{ [2 x target("dx.Layout", %Test, 8, 0, 4)], float, [3 x [2 x <2 x float>]], float,
// CHECK-SAME: target("dx.Layout", %anon, 4, 0), double, target("dx.Layout", %anon.0, 8, 0), float, <1 x double>, i16 }>

// CHECK: %Test = type <{ float, float }>
// CHECK: %anon = type <{ float }>
// CHECK: %anon.0 = type <{ <2 x i32> }>

// CHECK: %__cblayout_CB_A = type <{ [2 x double], [3 x <3 x float>], float, [3 x double], half, [1 x <2 x double>], float, [2 x <3 x half>], <3 x half> }>
// CHECK: %__cblayout_CB_B = type <{ [3 x <3 x double>], <3 x half> }>
// CHECK: %__cblayout_CB_C = type <{ i32, target("dx.Layout", %F, 96, 0, 16, 28, 32, 56, 64, 80, 84, 90), half, target("dx.Layout", %G, 258, 0, 48, 64, 256), double }>

// CHECK: %F = type <{ double, <3 x float>, float, <3 x double>, half, <2 x double>, float, <3 x half>, <3 x half> }>
// CHECK: %G = type <{ target("dx.Layout", %E, 36, 0, 8, 16, 20, 22, 24, 32), [1 x float], [2 x target("dx.Layout", %F, 96, 0, 16, 28, 32, 56, 64, 80, 84, 90)], half }>
// CHECK: %E = type <{ float, double, float, half, i16, i64, i32 }>

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

// CHECK: @CBScalars.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CBScalars, 
// CHECK-SAME: 56, 0, 8, 16, 24, 32, 36, 40, 48))
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
  // FIXME: add a bool vectors after llvm-project/llvm#91639 is added
}

// CHECK: @CBVectors.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CBVectors, 
// CHECK-SAME: 136, 0, 16, 40, 48, 80, 96, 112))
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

// CHECK: @CBArrays.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CBArrays, 
// CHECK-SAME: 708, 0, 48, 112, 176, 224, 608, 624, 656))
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

// CHECK: @CBStructs.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CBStructs, 
// CHECK-SAME: 246, 0, 16, 32, 64, 144, 238, 240))
// CHECK: @a = external addrspace(2) global target("dx.Layout", %A, 8, 0), align 8
// CHECK: @b = external addrspace(2) global target("dx.Layout", %B, 14, 0, 8), align 8
// CHECK: @c = external addrspace(2) global target("dx.Layout", %C, 24, 0, 16), align 8
// CHECK: @array_of_A = external addrspace(2) global [5 x target("dx.Layout", %A, 8, 0)], align 8
// CHECK: @d = external addrspace(2) global target("dx.Layout", %__cblayout_D, 94, 0), align 8
// CHECK: @e = external addrspace(2) global half, align 2
// CHECK: @f = external addrspace(2) global <3 x i16>, align 8

cbuffer CBStructs {
  A a;
  B b;
  C c;
  A array_of_A[5];
  D d;
  half e;
  uint16_t3 f;
};

struct Test {
    float a, b;
};

// CHECK: @CBMix.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CBMix,
// CHECK-SAME: 170, 0, 24, 32, 120, 128, 136, 144, 152, 160, 168))
// CHECK: @test = external addrspace(2) global [2 x target("dx.Layout", %Test, 8, 0, 4)], align 4
// CHECK: @f1 = external addrspace(2) global float, align 4
// CHECK: @f2 = external addrspace(2) global [3 x [2 x <2 x float>]], align 8
// CHECK: @f3 = external addrspace(2) global float, align 4
// CHECK: @f4 = external addrspace(2) global target("dx.Layout", %anon, 4, 0), align 4
// CHECK: @f5 = external addrspace(2) global double, align 8
// CHECK: @f6 = external addrspace(2) global target("dx.Layout", %anon.0, 8, 0), align 8
// CHECK: @f7 = external addrspace(2) global float, align 4
// CHECK: @f8 = external addrspace(2) global <1 x double>, align 8
// CHECK: @f9 = external addrspace(2) global i16, align 2

cbuffer CBMix {
    Test test[2];
    float f1;
    float2 f2[3][2];
    float f3;
    struct { float c; } f4;
    double f5;
    struct { int2 i; } f6;
    float f7;
    vector<double,1> f8;
    uint16_t f9;
};  

// CHECK: @CB_A.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_A, 188, 0, 32, 76, 80, 120, 128, 144, 160, 182))

cbuffer CB_A {
  double B0[2];
  float3 B1[3];
  float B2;
  double B3[3];
  half B4;
  double2 B5[1];
  float B6;
  half3 B7[2];
  half3 B8;
}

// CHECK: @CB_B.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_B, 94, 0, 88))
cbuffer CB_B {
  double3 B9[3];
  half3 B10;
}

struct E {
  float A0;
  double A1;
  float A2;
  half A3;
  int16_t A4;
  int64_t A5;
  int A6;
};

struct F {
  double B0;
  float3 B1;
  float B2;
  double3 B3;
  half B4;
  double2 B5;
  float B6;
  half3 B7;
  half3 B8;
};

struct G {
  E C0;
  float C1[1];
  F C2[2];
  half C3;
};

// CHECK: @CB_C.cb = external constant target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_C, 400, 0, 16, 112, 128, 392))
cbuffer CB_C {
  int D0;
  F D1;
  half D2;
  G D3;
  double D4;
}

// CHECK: define internal void @_init_resource_CBScalars.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[HANDLE1:.*]] = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBScalars, 56, 0, 8, 16, 24, 32, 36, 40, 48))
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBScalarss_56_0_8_16_24_32_36_40_48tt(i32 5, i32 1, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBScalars, 56, 0, 8, 16, 24, 32, 36, 40, 48)) %CBScalars.cb_h, ptr @CBScalars.cb, align 4

// CHECK: define internal void @_init_resource_CBArrays.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[HANDLE2:.*]] = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBArrays, 708, 0, 48, 112, 176, 224, 608, 624, 656))
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBArrayss_708_0_48_112_176_224_608_624_656tt(i32 0, i32 2, i32 1, i32 0, i1 false)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBArrays, 708, 0, 48, 112, 176, 224, 608, 624, 656)) %CBArrays.cb_h, ptr @CBArrays.cb, align 4

RWBuffer<float> Buf;

[numthreads(4,1,1)]
void main() {
  Buf[0] = a1 + b1.z + c1[2] + a.f1.y + f1 + B1[0].x + B10.z + D1.B2;
}

// CHECK: define internal void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_resource_CBScalars.cb()
// CHECK-NEXT: call void @_init_resource_CBArrays.cb()

// CHECK: !hlsl.cbs = !{![[CBSCALARS:[0-9]+]], ![[CBVECTORS:[0-9]+]], ![[CBARRAYS:[0-9]+]], ![[CBSTRUCTS:[0-9]+]], ![[CBMIX:[0-9]+]],
// CHECK-SAME: ![[CB_A:[0-9]+]], ![[CB_B:[0-9]+]], ![[CB_C:[0-9]+]]}

// CHECK: ![[CBSCALARS]] = !{ptr @CBScalars.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4,
// CHECK-SAME: ptr addrspace(2) @a5, ptr addrspace(2) @a6, ptr addrspace(2) @a7, ptr addrspace(2) @a8}

// CHECK: ![[CBVECTORS]] = !{ptr @CBVectors.cb, ptr addrspace(2) @b1, ptr addrspace(2) @b2, ptr addrspace(2) @b3, ptr addrspace(2) @b4,
// CHECK-SAME: ptr addrspace(2) @b5, ptr addrspace(2) @b6, ptr addrspace(2) @b7}

// CHECK: ![[CBARRAYS]] = !{ptr @CBArrays.cb, ptr addrspace(2) @c1, ptr addrspace(2) @c2, ptr addrspace(2) @c3, ptr addrspace(2) @c4, 
// CHECK-SAME: ptr addrspace(2) @c5, ptr addrspace(2) @c6, ptr addrspace(2) @c7, ptr addrspace(2) @c8}

// CHECK: ![[CBSTRUCTS]] = !{ptr @CBStructs.cb, ptr addrspace(2) @a, ptr addrspace(2) @b, ptr addrspace(2) @c, ptr addrspace(2) @array_of_A, 
// CHECK-SAME: ptr addrspace(2) @d, ptr addrspace(2) @e, ptr addrspace(2) @f}

// CHECK: ![[CBMIX]] = !{ptr @CBMix.cb, ptr addrspace(2) @test, ptr addrspace(2) @f1, ptr addrspace(2) @f2, ptr addrspace(2) @f3,
// CHECK-SAME: ptr addrspace(2) @f4, ptr addrspace(2) @f5, ptr addrspace(2) @f6, ptr addrspace(2) @f7, ptr addrspace(2) @f8, ptr addrspace(2) @f9}

// CHECK: ![[CB_A]] = !{ptr @CB_A.cb, ptr addrspace(2) @B0, ptr addrspace(2) @B1, ptr addrspace(2) @B2, ptr addrspace(2) @B3, ptr addrspace(2) @B4,
// CHECK-SAME: ptr addrspace(2) @B5, ptr addrspace(2) @B6, ptr addrspace(2) @B7, ptr addrspace(2) @B8}

// CHECK: ![[CB_B]] = !{ptr @CB_B.cb, ptr addrspace(2) @B9, ptr addrspace(2) @B10}

// CHECK: ![[CB_C]] = !{ptr @CB_C.cb, ptr addrspace(2) @D0, ptr addrspace(2) @D1, ptr addrspace(2) @D2, ptr addrspace(2) @D3, ptr addrspace(2) @D4}
