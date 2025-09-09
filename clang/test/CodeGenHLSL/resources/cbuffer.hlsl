// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -fnative-half-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

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

// CHECK: %__cblayout_CBClasses = type <{ target("dx.Layout", %K, 4, 0), target("dx.Layout", %L, 8, 0, 4),
// CHECK-SAME: target("dx.Layout", %M, 68, 0), [10 x target("dx.Layout", %K, 4, 0)] }>
// CHECK: %K = type <{ float }>
// CHECK: %L = type <{ float, float }>
// CHECK: %M = type <{ [5 x target("dx.Layout", %K, 4, 0)] }>

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

// CHECK: @CBScalars.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBScalars,
// CHECK-SAME: 56, 0, 8, 16, 24, 32, 36, 40, 48))
// CHECK: @a1 = external hidden addrspace(2) global float, align 4
// CHECK: @a2 = external hidden addrspace(2) global double, align 8
// CHECK: @a3 = external hidden addrspace(2) global half, align 2
// CHECK: @a4 = external hidden addrspace(2) global i64, align 8
// CHECK: @a5 = external hidden addrspace(2) global i32, align 4
// CHECK: @a6 = external hidden addrspace(2) global i16, align 2
// CHECK: @a7 = external hidden addrspace(2) global i32, align 4
// CHECK: @a8 = external hidden addrspace(2) global i64, align 8
// CHECK: @CBScalars.str = private unnamed_addr constant [10 x i8] c"CBScalars\00", align 1

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

// CHECK: @CBVectors.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBVectors,
// CHECK-SAME: 136, 0, 16, 40, 48, 80, 96, 112))
// CHECK: @b1 = external hidden addrspace(2) global <3 x float>, align 16
// CHECK: @b2 = external hidden addrspace(2) global <3 x double>, align 32
// CHECK: @b3 = external hidden addrspace(2) global <2 x half>, align 4
// CHECK: @b4 = external hidden addrspace(2) global <3 x i64>, align 32
// CHECK: @b5 = external hidden addrspace(2) global <4 x i32>, align 16
// CHECK: @b6 = external hidden addrspace(2) global <3 x i16>, align 8
// CHECK: @b7 = external hidden addrspace(2) global <3 x i64>, align 32
// CHECK: @CBVectors.str = private unnamed_addr constant [10 x i8] c"CBVectors\00", align 1

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

// CHECK: @CBArrays.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBArrays,
// CHECK-SAME: 708, 0, 48, 112, 176, 224, 608, 624, 656))
// CHECK: @c1 = external hidden addrspace(2) global [3 x float], align 4
// CHECK: @c2 = external hidden addrspace(2) global [2 x <3 x double>], align 32
// CHECK: @c3 = external hidden addrspace(2) global [2 x [2 x half]], align 2
// CHECK: @c4 = external hidden addrspace(2) global [3 x i64], align 8
// CHECK: @c5 = external hidden addrspace(2) global [2 x [3 x [4 x <4 x i32>]]], align 16
// CHECK: @c6 = external hidden addrspace(2) global [1 x i16], align 2
// CHECK: @c7 = external hidden addrspace(2) global [2 x i64], align 8
// CHECK: @c8 = external hidden addrspace(2) global [4 x i32], align 4
// CHECK: @CBArrays.str = private unnamed_addr constant [9 x i8] c"CBArrays\00", align 1

typedef uint32_t4 uint32_t8[2];
typedef uint4 T1;
typedef T1 T2[2]; // check a double typedef

cbuffer CBTypedefArray : register(space2) {
  uint32_t8 t1[2];
  T2 t2[2];
}

// CHECK: @CBTypedefArray.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBTypedefArray,
// CHECK-SAME: 128, 0, 64))
// CHECK: @t1 = external hidden addrspace(2) global [2 x [2 x <4 x i32>]], align 16
// CHECK: @t2 = external hidden addrspace(2) global [2 x [2 x <4 x i32>]], align 16
// CHECK: @CBTypedefArray.str = private unnamed_addr constant [15 x i8] c"CBTypedefArray\00", align 1
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

// CHECK: @CBStructs.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBStructs,
// CHECK-SAME: 246, 0, 16, 32, 64, 144, 238, 240))
// CHECK: @a = external hidden addrspace(2) global target("dx.Layout", %A, 8, 0), align 1
// CHECK: @b = external hidden addrspace(2) global target("dx.Layout", %B, 14, 0, 8), align 1
// CHECK: @c = external hidden addrspace(2) global target("dx.Layout", %C, 24, 0, 16), align 1
// CHECK: @array_of_A = external hidden addrspace(2) global [5 x target("dx.Layout", %A, 8, 0)], align 1
// CHECK: @d = external hidden addrspace(2) global target("dx.Layout", %__cblayout_D, 94, 0), align 1
// CHECK: @e = external hidden addrspace(2) global half, align 2
// CHECK: @f = external hidden addrspace(2) global <3 x i16>, align 8
// CHECK: @CBStructs.str = private unnamed_addr constant [10 x i8] c"CBStructs\00", align 1

cbuffer CBStructs {
  A a;
  B b;
  C c;
  A array_of_A[5];
  D d;
  half e;
  uint16_t3 f;
};


class K {
  float i;
};

class L : K {
  float j;
};

class M {
  K array[5];
};

cbuffer CBClasses {
  K k;
  L l;
  M m;
  K ka[10];
};

// CHECK: @CBClasses.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBClasses,
// CHECK-SAME: 260, 0, 16, 32, 112))
// CHECK: @k = external hidden addrspace(2) global target("dx.Layout", %K, 4, 0), align 1
// CHECK: @l = external hidden addrspace(2) global target("dx.Layout", %L, 8, 0, 4), align 1
// CHECK: @m = external hidden addrspace(2) global target("dx.Layout", %M, 68, 0), align 1
// CHECK: @ka = external hidden addrspace(2) global [10 x target("dx.Layout", %K, 4, 0)], align 1
// CHECK: @CBClasses.str = private unnamed_addr constant [10 x i8] c"CBClasses\00", align 1

struct Test {
    float a, b;
};

// CHECK: @CBMix.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CBMix,
// CHECK-SAME: 170, 0, 24, 32, 120, 128, 136, 144, 152, 160, 168))
// CHECK: @test = external hidden addrspace(2) global [2 x target("dx.Layout", %Test, 8, 0, 4)], align 1
// CHECK: @f1 = external hidden addrspace(2) global float, align 4
// CHECK: @f2 = external hidden addrspace(2) global [3 x [2 x <2 x float>]], align 8
// CHECK: @f3 = external hidden addrspace(2) global float, align 4
// CHECK: @f4 = external hidden addrspace(2) global target("dx.Layout", %anon, 4, 0), align 1
// CHECK: @f5 = external hidden addrspace(2) global double, align 8
// CHECK: @f6 = external hidden addrspace(2) global target("dx.Layout", %anon.0, 8, 0), align 1
// CHECK: @f7 = external hidden addrspace(2) global float, align 4
// CHECK: @f8 = external hidden addrspace(2) global <1 x double>, align 8
// CHECK: @f9 = external hidden addrspace(2) global i16, align 2
// CHECK: @CBMix.str = private unnamed_addr constant [6 x i8] c"CBMix\00", align 1

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

// CHECK: @CB_A.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_A, 188, 0, 32, 76, 80, 120, 128, 144, 160, 182))

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

// CHECK: @CB_B.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_B, 94, 0, 88))
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

// CHECK: @CB_C.cb = global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_C, 400, 0, 16, 112, 128, 392))
cbuffer CB_C {
  int D0;
  F D1;
  half D2;
  G D3;
  double D4;
}

// CHECK: define internal void @_init_buffer_CBScalars.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBScalars.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBScalars, 56, 0, 8, 16, 24, 32, 36, 40, 48))
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBScalarss_56_0_8_16_24_32_36_40_48tt(i32 5, i32 1, i32 1, i32 0, ptr @CBScalars.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBScalars, 56, 0, 8, 16, 24, 32, 36, 40, 48)) %CBScalars.cb_h, ptr @CBScalars.cb, align 4

// CHECK: define internal void @_init_buffer_CBVectors.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBVectors.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBVectors, 136, 0, 16, 40, 48, 80, 96, 112))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBVectorss_136_0_16_40_48_80_96_112tt(i32 0, i32 0, i32 1, i32 0, ptr @CBVectors.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBVectors, 136, 0, 16, 40, 48, 80, 96, 112)) %CBVectors.cb_h, ptr @CBVectors.cb, align 4

// CHECK: define internal void @_init_buffer_CBArrays.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBArrays.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBArrays, 708, 0, 48, 112, 176, 224, 608, 624, 656))
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBArrayss_708_0_48_112_176_224_608_624_656tt(i32 0, i32 2, i32 1, i32 0, ptr @CBArrays.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBArrays, 708, 0, 48, 112, 176, 224, 608, 624, 656)) %CBArrays.cb_h, ptr @CBArrays.cb, align 4

// CHECK: define internal void @_init_buffer_CBTypedefArray.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBTypedefArray.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBTypedefArray, 128, 0, 64))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBTypedefArrays_128_0_64tt(i32 1, i32 2, i32 1, i32 0, ptr @CBTypedefArray.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBTypedefArray, 128, 0, 64)) %CBTypedefArray.cb_h, ptr @CBTypedefArray.cb, align 4

// CHECK: define internal void @_init_buffer_CBStructs.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %CBStructs.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBStructs, 246, 0, 16, 32, 64, 144, 238, 240))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBStructss_246_0_16_32_64_144_238_240tt(i32 2, i32 0, i32 1, i32 0, ptr @CBStructs.str)
// CHECK-NEXT:   store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBStructs, 246, 0, 16, 32, 64, 144, 238, 240)) %CBStructs.cb_h, ptr @CBStructs.cb, align 4

// CHECK: define internal void @_init_buffer_CBClasses.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBClasses.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBClasses, 260, 0, 16, 32, 112))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBClassess_260_0_16_32_112tt(i32 3, i32 0, i32 1, i32 0, ptr @CBClasses.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBClasses, 260, 0, 16, 32, 112)) %CBClasses.cb_h, ptr @CBClasses.cb, align 4

// CHECK: define internal void @_init_buffer_CBMix.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBMix.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CBMix, 170, 0, 24, 32, 120, 128, 136, 144, 152, 160, 168))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBMixs_170_0_24_32_120_128_136_144_152_160_168tt(i32 4, i32 0, i32 1, i32 0, ptr @CBMix.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CBMix, 170, 0, 24, 32, 120, 128, 136, 144, 152, 160, 168)) %CBMix.cb_h, ptr @CBMix.cb, align 4

// CHECK: define internal void @_init_buffer_CB_A.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_A.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_A, 188, 0, 32, 76, 80, 120, 128, 144, 160, 182))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CB_As_188_0_32_76_80_120_128_144_160_182tt(i32 5, i32 0, i32 1, i32 0, ptr @CB_A.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_A, 188, 0, 32, 76, 80, 120, 128, 144, 160, 182)) %CB_A.cb_h, ptr @CB_A.cb, align 4

// CHECK: define internal void @_init_buffer_CB_B.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_B.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_B, 94, 0, 88))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CB_Bs_94_0_88tt(i32 6, i32 0, i32 1, i32 0, ptr @CB_B.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_B, 94, 0, 88)) %CB_B.cb_h, ptr @CB_B.cb, align 4

// CHECK: define internal void @_init_buffer_CB_C.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_C.cb_h = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_C, 400, 0, 16, 112, 128, 392))
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_tdx.Layout_s___cblayout_CB_Cs_400_0_16_112_128_392tt(i32 7, i32 0, i32 1, i32 0, ptr @CB_C.str)
// CHECK-NEXT: store target("dx.CBuffer", target("dx.Layout", %__cblayout_CB_C, 400, 0, 16, 112, 128, 392)) %CB_C.cb_h, ptr @CB_C.cb, align 4

RWBuffer<float> Buf;

[numthreads(4,1,1)]
void main() {
  Buf[0] = a1 + b1.z + c1[2] + a.f1.y + f1 + B1[0].x + ka[2].i + B10.z + D1.B2;
}

// CHECK: define internal void @_GLOBAL__sub_I_cbuffer.hlsl()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @_init_buffer_CBScalars.cb()
// CHECK-NEXT: call void @_init_buffer_CBVectors.cb()
// CHECK-NEXT: call void @_init_buffer_CBArrays.cb()
// CHECK-NEXT: call void @_init_buffer_CBTypedefArray.cb()
// CHECK-NEXT: call void @_init_buffer_CBStructs.cb()
// CHECK-NEXT: call void @_init_buffer_CBClasses.cb()
// CHECK-NEXT: call void @_init_buffer_CBMix.cb()
// CHECK-NEXT: call void @_init_buffer_CB_A.cb()

// CHECK: !hlsl.cbs = !{![[CBSCALARS:[0-9]+]], ![[CBVECTORS:[0-9]+]], ![[CBARRAYS:[0-9]+]], ![[CBTYPEDEFARRAY:[0-9]+]], ![[CBSTRUCTS:[0-9]+]], ![[CBCLASSES:[0-9]+]],
// CHECK-SAME: ![[CBMIX:[0-9]+]], ![[CB_A:[0-9]+]], ![[CB_B:[0-9]+]], ![[CB_C:[0-9]+]]}

// CHECK: ![[CBSCALARS]] = !{ptr @CBScalars.cb, ptr addrspace(2) @a1, ptr addrspace(2) @a2, ptr addrspace(2) @a3, ptr addrspace(2) @a4,
// CHECK-SAME: ptr addrspace(2) @a5, ptr addrspace(2) @a6, ptr addrspace(2) @a7, ptr addrspace(2) @a8}

// CHECK: ![[CBVECTORS]] = !{ptr @CBVectors.cb, ptr addrspace(2) @b1, ptr addrspace(2) @b2, ptr addrspace(2) @b3, ptr addrspace(2) @b4,
// CHECK-SAME: ptr addrspace(2) @b5, ptr addrspace(2) @b6, ptr addrspace(2) @b7}

// CHECK: ![[CBARRAYS]] = !{ptr @CBArrays.cb, ptr addrspace(2) @c1, ptr addrspace(2) @c2, ptr addrspace(2) @c3, ptr addrspace(2) @c4,
// CHECK-SAME: ptr addrspace(2) @c5, ptr addrspace(2) @c6, ptr addrspace(2) @c7, ptr addrspace(2) @c8}

// CHECK: ![[CBTYPEDEFARRAY]] = !{ptr @CBTypedefArray.cb, ptr addrspace(2) @t1, ptr addrspace(2) @t2}

// CHECK: ![[CBSTRUCTS]] = !{ptr @CBStructs.cb, ptr addrspace(2) @a, ptr addrspace(2) @b, ptr addrspace(2) @c, ptr addrspace(2) @array_of_A,
// CHECK-SAME: ptr addrspace(2) @d, ptr addrspace(2) @e, ptr addrspace(2) @f}

// CHECK: ![[CBCLASSES]] = !{ptr @CBClasses.cb, ptr addrspace(2) @k, ptr addrspace(2) @l, ptr addrspace(2) @m, ptr addrspace(2) @ka}

// CHECK: ![[CBMIX]] = !{ptr @CBMix.cb, ptr addrspace(2) @test, ptr addrspace(2) @f1, ptr addrspace(2) @f2, ptr addrspace(2) @f3,
// CHECK-SAME: ptr addrspace(2) @f4, ptr addrspace(2) @f5, ptr addrspace(2) @f6, ptr addrspace(2) @f7, ptr addrspace(2) @f8, ptr addrspace(2) @f9}

// CHECK: ![[CB_A]] = !{ptr @CB_A.cb, ptr addrspace(2) @B0, ptr addrspace(2) @B1, ptr addrspace(2) @B2, ptr addrspace(2) @B3, ptr addrspace(2) @B4,
// CHECK-SAME: ptr addrspace(2) @B5, ptr addrspace(2) @B6, ptr addrspace(2) @B7, ptr addrspace(2) @B8}

// CHECK: ![[CB_B]] = !{ptr @CB_B.cb, ptr addrspace(2) @B9, ptr addrspace(2) @B10}

// CHECK: ![[CB_C]] = !{ptr @CB_C.cb, ptr addrspace(2) @D0, ptr addrspace(2) @D1, ptr addrspace(2) @D2, ptr addrspace(2) @D3, ptr addrspace(2) @D4}
