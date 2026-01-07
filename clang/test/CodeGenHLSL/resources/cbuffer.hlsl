// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: %__cblayout_CBScalars = type <{
// CHECK-SAME:   float, target("dx.Padding", 4), double,
// CHECK-SAME:   half, target("dx.Padding", 6), i64,
// CHECK-SAME:   i32, i16, target("dx.Padding", 2), i32, target("dx.Padding", 4),
// CHECK-SAME:   i64
// CHECK-SAME: }>

// CHECK: %__cblayout_CBVectors = type <{
// CHECK-SAME:   <3 x float>, target("dx.Padding", 4),
// CHECK-SAME:   <3 x double>, <2 x half>, target("dx.Padding", 4),
// CHECK-SAME:   <3 x i64>, target("dx.Padding", 8),
// CHECK-SAME:   <4 x i32>,
// CHECK-SAME:   <3 x i16>, target("dx.Padding", 10),
// CHECK-SAME:   <3 x i64>
// CHECK-SAME: }>

// CHECK: %__cblayout_CBArrays = type <{
// CHECK-SAME: <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, target("dx.Padding", 12),
// CHECK-SAME: <{ [1 x <{ <3 x double>, target("dx.Padding", 8) }>], <3 x double> }>, target("dx.Padding", 8),
// CHECK-SAME: <{ [1 x <{
// CHECK-SAME:   <{ [1 x <{ half, target("dx.Padding", 14) }>], half }>, target("dx.Padding", 14) }>],
// CHECK-SAME:   <{ [1 x <{ half, target("dx.Padding", 14) }>], half }>
// CHECK-SAME: }>, target("dx.Padding", 14),
// CHECK-SAME: <{ [2 x <{ i64, target("dx.Padding", 8) }>], i64 }>, target("dx.Padding", 8),
// CHECK-SAME: [2 x [3 x [4 x <4 x i32>]]]
// CHECK-SAME: [1 x i16], target("dx.Padding", 14),
// CHECK-SAME: <{ [1 x <{ i64, target("dx.Padding", 8) }>], i64 }>, target("dx.Padding", 8),
// CHECK-SAME: <{ [3 x <{ i32, target("dx.Padding", 12) }>], i32 }>
// CHECK-SAME: }>

// CHECK: %__cblayout_CBStructs = type <{
// CHECK-SAME:   %A, target("dx.Padding", 8),

// TODO: We should have target("dx.Padding", 2) padding after %B, but we don't
// correctly handle 2- and 3-element vectors inside structs yet because of
// DataLayout rules. See https://github.com/llvm/llvm-project/issues/123968.
//
// CHECK-SAME: %B,

// CHECK-SAME:   %C, target("dx.Padding", 8),
// CHECK-SAME:   <{ [4 x <{ %A, target("dx.Padding", 8) }>], %A }>, target("dx.Padding", 8),
// CHECK-SAME:   %__cblayout_D, half,
// CHECK-SAME:   <3 x i16>
// CHECK-SAME: }>

// CHECK: %A = type <{ <2 x float> }>
// CHECK: %B = type <{ <2 x float>, <3 x i16> }>
// CHECK: %C = type <{ i32, target("dx.Padding", 12), %A }>

// CHECK: %__cblayout_D = type <{
// CHECK-SAME:   <{ [1 x <{
// CHECK-SAME:     <{ [2 x <{ %B, target("dx.Padding", 2) }>], %B }>, target("dx.Padding", 2)
// CHECK-SAME:   }>],
// CHECK-SAME:   <{ [2 x <{ %B, target("dx.Padding", 2) }>], %B }> }>
// CHECK-SAME: }>

// CHECK: %__cblayout_CBClasses = type <{
// CHECK-SAME:   %K, target("dx.Padding", 12),
// CHECK-SAME:   %L, target("dx.Padding", 8),
// CHECK-SAME:   %M, target("dx.Padding", 12),
// CHECK-SAME:   <{ [9 x <{ %K, target("dx.Padding", 12) }>], %K }>
// CHECK-SAME: }>

// CHECK: %K = type <{ float }>
// CHECK: %L = type <{ float, float }>
// CHECK: %M = type <{ <{ [4 x <{ %K, target("dx.Padding", 12) }>], %K }> }>

// CHECK: %__cblayout_CBMix = type <{
// CHECK-SAME:   <{ [1 x <{ %Test, target("dx.Padding", 8) }>], %Test }>, float, target("dx.Padding", 4)
// CHECK-SAME:   <{ [2 x <{
// CHECK-SAME:     <{ [1 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>, target("dx.Padding", 8) }>],
// CHECK-SAME:     <{ [1 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>
// CHECK-SAME:   }>, float, target("dx.Padding", 4),
// CHECK-SAME:   %anon, target("dx.Padding", 4), double,
// CHECK-SAME:   %anon.0, float, target("dx.Padding", 4),
// CHECK-SAME:   <1 x double>, i16
// CHECK-SAME: }>

// CHECK: %Test = type <{ float, float }>
// CHECK: %anon = type <{ float }>
// CHECK: %anon.0 = type <{ <2 x i32> }>

// CHECK: %__cblayout_CB_A = type <{
// CHECK-SAME:   <{ [1 x <{ double, target("dx.Padding", 8) }>], double }>, target("dx.Padding", 8),
// CHECK-SAME:   <{ [2 x <{ <3 x float>, target("dx.Padding", 4) }>], <3 x float> }>, float,
// CHECK-SAME:   <{ [2 x <{ double, target("dx.Padding", 8) }>], double }>, half, target("dx.Padding", 6),
// CHECK-SAME:   [1 x <2 x double>],
// CHECK-SAME:   float, target("dx.Padding", 12),
// CHECK-SAME:   <{ [1 x <{ <3 x half>, target("dx.Padding", 10) }>], <3 x half> }>, <3 x half>
// CHECK-SAME: }>

// CHECK: %__cblayout_CB_B = type <{
// CHECK-SAME: <{ [2 x <{ <3 x double>, target("dx.Padding", 8) }>], <3 x double> }>, <3 x half>
// CHECK-SAME: }>

// CHECK: %__cblayout_CB_C = type <{
// CHECK-SAME:   i32, target("dx.Padding", 12),
// CHECK-SAME:   %F,
// CHECK-SAME:   half, target("dx.Padding", 14),
// CHECK-SAME:   %G, target("dx.Padding", 6), double
// CHECK-SAME: }>

// CHECK: %F = type <{
// CHECK-SAME:   double, target("dx.Padding", 8),
// CHECK-SAME:   <3 x float>, float,
// CHECK-SAME:   <3 x double>, half, target("dx.Padding", 6),
// CHECK-SAME:   <2 x double>,
// CHECK-SAME:   float, <3 x half>, <3 x half>
// CHECK-SAME: }>

// CHECK: %G = type <{
// CHECK-SAME:   %E, target("dx.Padding", 12),
// CHECK-SAME:   [1 x float], target("dx.Padding", 12),
// CHECK-SAME:   [2 x %F],
// CHECK-SAME:   half
// CHECK-SAME: }>

// CHECK: %E = type <{ float, target("dx.Padding", 4), double, float, half, i16, i64, i32 }>

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

// CHECK: @CBScalars.cb = global target("dx.CBuffer", %__cblayout_CBScalars)
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

// CHECK: @CBVectors.cb = global target("dx.CBuffer", %__cblayout_CBVectors)
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

// CHECK: @CBArrays.cb = global target("dx.CBuffer", %__cblayout_CBArrays)
// CHECK: @c1 = external hidden addrspace(2) global <{ [2 x <{ float, target("dx.Padding", 12) }>], float }>, align 4
// CHECK: @c2 = external hidden addrspace(2) global <{ [1 x <{ <3 x double>, target("dx.Padding", 8) }>], <3 x double> }>, align 32
// CHECK: @c3 = external hidden addrspace(2) global <{ [1 x <{ <{ [1 x <{ half, target("dx.Padding", 14) }>], half }>, target("dx.Padding", 14) }>], <{ [1 x <{ half, target("dx.Padding", 14) }>], half }> }>, align 2
// CHECK: @c4 = external hidden addrspace(2) global <{ [2 x <{ i64, target("dx.Padding", 8) }>], i64 }>, align 8
// CHECK: @c5 = external hidden addrspace(2) global [2 x [3 x [4 x <4 x i32>]]], align 16
// CHECK: @c6 = external hidden addrspace(2) global [1 x i16], align 2
// CHECK: @c7 = external hidden addrspace(2) global <{ [1 x <{ i64, target("dx.Padding", 8) }>], i64 }>, align 8
// CHECK: @c8 = external hidden addrspace(2) global <{ [3 x <{ i32, target("dx.Padding", 12) }>], i32 }>, align 4
// CHECK: @CBArrays.str = private unnamed_addr constant [9 x i8] c"CBArrays\00", align 1

typedef uint32_t4 uint32_t8[2];
typedef uint4 T1;
typedef T1 T2[2]; // check a double typedef

cbuffer CBTypedefArray : register(space2) {
  uint32_t8 t1[2];
  T2 t2[2];
}

// CHECK: @CBTypedefArray.cb = global target("dx.CBuffer", %__cblayout_CBTypedefArray)
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

// CHECK: @CBStructs.cb = global target("dx.CBuffer", %__cblayout_CBStructs)
// CHECK: @a = external hidden addrspace(2) global %A, align 1
// CHECK: @b = external hidden addrspace(2) global %B, align 1
// CHECK: @c = external hidden addrspace(2) global %C, align 1
// CHECK: @array_of_A = external hidden addrspace(2) global <{ [4 x <{ %A, target("dx.Padding", 8) }>], %A }>, align 1
// CHECK: @d = external hidden addrspace(2) global %__cblayout_D, align 1
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

// CHECK: @CBClasses.cb = global target("dx.CBuffer", %__cblayout_CBClasses)
// CHECK: @k = external hidden addrspace(2) global %K, align 1
// CHECK: @l = external hidden addrspace(2) global %L, align 1
// CHECK: @m = external hidden addrspace(2) global %M, align 1
// CHECK: @ka = external hidden addrspace(2) global <{ [9 x <{ %K, target("dx.Padding", 12) }>], %K }>, align 1
// CHECK: @CBClasses.str = private unnamed_addr constant [10 x i8] c"CBClasses\00", align 1

struct Test {
    float a, b;
};

// CHECK: @CBMix.cb = global target("dx.CBuffer", %__cblayout_CBMix)
// CHECK: @test = external hidden addrspace(2) global <{ [1 x <{ %Test, target("dx.Padding", 8) }>], %Test }>, align 1
// CHECK: @f1 = external hidden addrspace(2) global float, align 4
// CHECK: @f2 = external hidden addrspace(2) global <{ [2 x <{ <{ [1 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>, target("dx.Padding", 8) }>], <{ [1 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }> }>, align 8
// CHECK: @f3 = external hidden addrspace(2) global float, align 4
// CHECK: @f4 = external hidden addrspace(2) global %anon, align 1
// CHECK: @f5 = external hidden addrspace(2) global double, align 8
// CHECK: @f6 = external hidden addrspace(2) global %anon.0, align 1
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

// CHECK: @CB_A.cb = global target("dx.CBuffer", %__cblayout_CB_A)

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

// CHECK: @CB_B.cb = global target("dx.CBuffer", %__cblayout_CB_B)
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

// CHECK: @CB_C.cb = global target("dx.CBuffer", %__cblayout_CB_C)
cbuffer CB_C {
  int D0;
  F D1;
  half D2;
  G D3;
  double D4;
}

// CHECK: define internal void @_init_buffer_CBScalars.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBScalars.cb_h = call target("dx.CBuffer", %__cblayout_CBScalars)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_CBScalarsst(i32 5, i32 1, i32 1, i32 0, ptr @CBScalars.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CBScalars) %CBScalars.cb_h, ptr @CBScalars.cb, align 4

// CHECK: define internal void @_init_buffer_CBVectors.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBVectors.cb_h = call target("dx.CBuffer", %__cblayout_CBVectors)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBVectorsst(i32 0, i32 0, i32 1, i32 0, ptr @CBVectors.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CBVectors) %CBVectors.cb_h, ptr @CBVectors.cb, align 4

// CHECK: define internal void @_init_buffer_CBArrays.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBArrays.cb_h = call target("dx.CBuffer", %__cblayout_CBArrays)
// CHECK-SAME: @llvm.dx.resource.handlefrombinding.tdx.CBuffer_s___cblayout_CBArraysst(i32 0, i32 2, i32 1, i32 0, ptr @CBArrays.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CBArrays) %CBArrays.cb_h, ptr @CBArrays.cb, align 4

// CHECK: define internal void @_init_buffer_CBTypedefArray.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBTypedefArray.cb_h = call target("dx.CBuffer", %__cblayout_CBTypedefArray)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBTypedefArrayst(i32 1, i32 2, i32 1, i32 0, ptr @CBTypedefArray.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CBTypedefArray) %CBTypedefArray.cb_h, ptr @CBTypedefArray.cb, align 4

// CHECK: define internal void @_init_buffer_CBStructs.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %CBStructs.cb_h = call target("dx.CBuffer", %__cblayout_CBStructs)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBStructsst(i32 2, i32 0, i32 1, i32 0, ptr @CBStructs.str)
// CHECK-NEXT:   store target("dx.CBuffer", %__cblayout_CBStructs) %CBStructs.cb_h, ptr @CBStructs.cb, align 4

// CHECK: define internal void @_init_buffer_CBClasses.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBClasses.cb_h = call target("dx.CBuffer", %__cblayout_CBClasses)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBClassesst(i32 3, i32 0, i32 1, i32 0, ptr @CBClasses.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CBClasses) %CBClasses.cb_h, ptr @CBClasses.cb, align 4

// CHECK: define internal void @_init_buffer_CBMix.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CBMix.cb_h = call target("dx.CBuffer", %__cblayout_CBMix)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBMixst(i32 4, i32 0, i32 1, i32 0, ptr @CBMix.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CBMix) %CBMix.cb_h, ptr @CBMix.cb, align 4

// CHECK: define internal void @_init_buffer_CB_A.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_A.cb_h = call target("dx.CBuffer", %__cblayout_CB_A)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CB_Ast(i32 5, i32 0, i32 1, i32 0, ptr @CB_A.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CB_A) %CB_A.cb_h, ptr @CB_A.cb, align 4

// CHECK: define internal void @_init_buffer_CB_B.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_B.cb_h = call target("dx.CBuffer", %__cblayout_CB_B)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CB_Bst(i32 6, i32 0, i32 1, i32 0, ptr @CB_B.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CB_B) %CB_B.cb_h, ptr @CB_B.cb, align 4

// CHECK: define internal void @_init_buffer_CB_C.cb()
// CHECK-NEXT: entry:
// CHECK-NEXT: %CB_C.cb_h = call target("dx.CBuffer", %__cblayout_CB_C)
// CHECK-SAME: @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CB_Cst(i32 7, i32 0, i32 1, i32 0, ptr @CB_C.str)
// CHECK-NEXT: store target("dx.CBuffer", %__cblayout_CB_C) %CB_C.cb_h, ptr @CB_C.cb, align 4

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
