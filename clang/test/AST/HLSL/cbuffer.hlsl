// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump -o - %s | FileCheck %s

struct EmptyStruct {
};

struct A {
  float a;
};

struct B {
  RWBuffer<float> buf;
  EmptyStruct es;
  float ea[0];
  float a;
};

struct C {
  EmptyStruct es;
};

typedef B BTypedef;
typedef C CTypedef;

struct D : B {
  float b;
};

struct E : EmptyStruct {
  float c;
};

struct F : A {
  int ae[0];
};

typedef float EmptyArrayTypedef[10][0];

struct OneFloat {
  float a;
};

struct TwoFloats {
  float a;
  float b;
};

// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} used a1 'hlsl_constant float'
  float a1;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_CB definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} a1 'float'
}
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_CB), "");

// Check that buffer layout struct does not include resources or empty types 
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} used a2 'hlsl_constant float'
  float a2;
  // CHECK: VarDecl {{.*}} b2 'RWBuffer<float>':'hlsl::RWBuffer<float>'
  RWBuffer<float> b2; 
  // CHECK: VarDecl {{.*}} c2 'EmptyStruct'
  EmptyStruct c2;
  // CHECK: VarDecl {{.*}} d2 'float[0]'
  float d2[0];
  // CHECK: VarDecl {{.*}} f2 'RWBuffer<float>[2]'
  RWBuffer<float> f2[2];
  // CHECK: VarDecl {{.*}} g2 'groupshared float'
  groupshared float g2;
  // CHECK: VarDecl {{.*}} h2 '__hlsl_resource_t'
  __hlsl_resource_t h2;
  // CHECK: VarDecl {{.*}} e2 'hlsl_constant float'
  float e2;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_CB_1 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} a2 'float'
  // CHECK-NEXT: FieldDecl {{.*}} e2 'float'
}
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __cblayout_CB_1), "");

// Check that layout struct is created for B and the empty struct C is removed
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} used s1 'hlsl_constant A'
  A s1;
  // CHECK: VarDecl {{.*}} s2 'hlsl_constant B'
  B s2;
  // CHECK: VarDecl {{.*}} s3 'CTypedef':'C'
  CTypedef s3;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_CB_2 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} s1 'A'
  // CHECK-NEXT: FieldDecl {{.*}} s2 '__cblayout_B'
}
// CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_B definition
// CHECK: PackedAttr
// CHECK-NEXT: FieldDecl {{.*}} a 'float'

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_B), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __cblayout_CB_2), "");

// check that layout struct is created for D because of its base struct
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} s4 'hlsl_constant D'
  D s4;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_CB_3 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} s4 '__cblayout_D'
}
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_D definition
  // CHECK: public '__cblayout_B'
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} b 'float'
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __cblayout_D), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __cblayout_CB_3), "");

// check that layout struct is created for E because because its base struct
// is empty and should be eliminated, and BTypedef should reuse the previously
// defined '__cblayout_B' 
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}}  s5 'hlsl_constant E'
  E s5;
  // CHECK: VarDecl {{.*}} s6 'hlsl_constant BTypedef':'hlsl_constant B'
  BTypedef s6;
  // CHECK: CXXRecordDecl {{.*}}  implicit referenced struct __cblayout_CB_4 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} s5 '__cblayout_E'
  // CHECK-NEXT: FieldDecl {{.*}} s6 '__cblayout_B'
}
// CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_E definition
// CHECK: PackedAttr
// CHECK-NEXT: FieldDecl {{.*}} c 'float'
// CHECK-NOT: CXXRecordDecl {{.*}} struct __cblayout_B definition
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_E), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __cblayout_CB_4), "");

// check that this produces empty layout struct
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: FunctionDecl {{.*}} f 'void ()'
  void f() {}
  // CHECK: VarDecl {{.*}} SV 'hlsl_private float' static
  static float SV;
  // CHECK: VarDecl {{.*}} s7 'EmptyStruct' callinit
  EmptyStruct s7;
  // CHECK: VarDecl {{.*}} Buf 'RWBuffer<float>':'hlsl::RWBuffer<float>' static callinit
  RWBuffer<float> Buf;
  // CHECK: VarDecl {{.*}} ea 'EmptyArrayTypedef':'float[10][0]'
  EmptyArrayTypedef ea;
  // CHECK: CXXRecordDecl {{.*}} implicit struct __cblayout_CB_5 definition
  // CHECK: PackedAttr
  // CHECK-NOT: FieldDecl
}

// check host layout struct with compatible base struct
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} s8 'hlsl_constant F'
  F s8;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_CB_6 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} s8 '__cblayout_F'
}
// CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_F definition
// CHECK: public 'A'
// CHECK: PackedAttr
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_F), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_CB_6), "");

// anonymous structs
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 2]]:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: CXXRecordDecl {{.*}} struct definition
  struct {
    // CHECK: FieldDecl {{.*}} e 'float'
    float e;
    // CHECK: FieldDecl {{.*}} c 'int[0][1]'
    int c[0][1];
    // CHECK: FieldDecl {{.*}} f 'RWBuffer<float>':'hlsl::RWBuffer<float>'
    RWBuffer<float> f;
  } s9;
  // CHECK: VarDecl {{.*}} s9 'hlsl_constant struct (unnamed struct at {{.*}}cbuffer.hlsl:[[# @LINE - 8]]:3
  // CHECK: CXXRecordDecl {{.*}} struct definition
  struct {
    // CHECK: FieldDecl {{.*}} g 'float'
    float g;
    // CHECK: FieldDecl {{.*}} f 'RWBuffer<float>':'hlsl::RWBuffer<float>'
    RWBuffer<float> f;
  } s10;
  // CHECK: VarDecl {{.*}} s10 'hlsl_constant struct (unnamed struct at {{.*}}cbuffer.hlsl:[[# @LINE - 6]]:3
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_anon definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} e 'float'
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_anon_1 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} g 'float'
  // CHECK: CXXRecordDecl {{.*}} implicit referenced struct __cblayout_CB_7 definition
  // CHECK: PackedAttr
  // CHECK-NEXT: FieldDecl {{.*}} s9 '__cblayout_anon'
  // CHECK-NEXT: FieldDecl {{.*}} s10 '__cblayout_anon_1'
}
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_anon), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __cblayout_anon_1), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __cblayout_CB_7), "");

// Add uses for the constant buffer declarations so they are not optimized away
export float foo() {
  return a1 + a2 + s1.a + s4.b + s5.c + s8.a + s9.e;
}
