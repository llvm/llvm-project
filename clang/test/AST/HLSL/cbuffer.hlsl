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

// CHECK: HLSLBufferDecl {{.*}} line:50:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} col:9 used a1 'hlsl_constant float'
  float a1;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB definition
  // CHECK: FieldDecl {{.*}} a1 'float'
}
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_CB), "");

// Check that buffer layout struct does not include resources or empty types 
// CHECK: HLSLBufferDecl {{.*}} line:62:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} col:9 used a2 'hlsl_constant float'
  float a2;
  // CHECK: VarDecl {{.*}} col:19 b2 'RWBuffer<float>':'hlsl::RWBuffer<float>'
  RWBuffer<float> b2; 
  // CHECK: VarDecl {{.*}} col:15 c2 'EmptyStruct'
  EmptyStruct c2;
  // CHECK: VarDecl {{.*}} col:9 d2 'float[0]'
  float d2[0];
  // CHECK: VarDecl {{.*}} col:9 e2 'hlsl_constant float'
  float e2;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB_1 definition
  // CHECK: FieldDecl {{.*}} a2 'float'
  // CHECK-NEXT: FieldDecl {{.*}} e2 'float'
}
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __layout_CB_1), "");

// Check that layout struct is created for B and the empty struct C is removed
// CHECK: HLSLBufferDecl {{.*}} line:83:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} col:5 used s1 'hlsl_constant A'
  A s1;
  // CHECK: VarDecl {{.*}} col:5 s2 'hlsl_constant B'
  B s2;
  // CHECK: VarDecl {{.*}} col:12 s3 'CTypedef':'C'
  CTypedef s3;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB_2 definition
  // CHECK: FieldDecl {{.*}} s1 'A'
  // CHECK: FieldDecl {{.*}} s2 '__layout_B'
}
// CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_B definition
// CHECK: FieldDecl {{.*}} a 'float'

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_B), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __layout_CB_2), "");

// check that layout struct is created for D because of its base struct
// CHECK: HLSLBufferDecl {{.*}} line:104:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} s4 'hlsl_constant D'
  D s4;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB_3 definition
  // CHECK: FieldDecl {{.*}} s4 '__layout_D'
}
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_D definition
  // CHECK: public '__layout_B'
  // CHECK: FieldDecl {{.*}} b 'float'
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __layout_D), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __layout_CB_3), "");

// check that layout struct is created for E because because its base struct
// is empty and should be eliminated, and BTypedef should reuse the previously
// defined '__layout_B' 
// CHECK: HLSLBufferDecl {{.*}} line:122:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}}  s5 'hlsl_constant E'
  E s5;
  // CHECK: VarDecl {{.*}} s6 'hlsl_constant BTypedef':'hlsl_constant B'
  BTypedef s6;
  // CHECK: CXXRecordDecl {{.*}}  implicit referenced class __layout_CB_4 definition
  // CHECK: FieldDecl {{.*}} s5 '__layout_E'
  // CHECK: FieldDecl {{.*}} s6 '__layout_B'
}
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_E definition
  // CHECK: FieldDecl {{.*}} c 'float'
  // CHECK-NOT: CXXRecordDecl {{.*}} class __layout_B definition
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_E), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __layout_CB_4), "");

// check that this produces empty layout struct
// CHECK: HLSLBufferDecl {{.*}} line:141:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: FunctionDecl {{.*}} f 'void ()'
  void f() {}
  // CHECK: VarDecl {{.*}} SV 'float' static
  static float SV;
  // CHECK: VarDecl {{.*}} s7 'EmptyStruct' callinit
  EmptyStruct s7;
  // CHECK: VarDecl {{.*}} Buf 'RWBuffer<float>':'hlsl::RWBuffer<float>' callinit
  RWBuffer<float> Buf;
  // CHECK: VarDecl {{.*}} ea 'EmptyArrayTypedef':'float[10][0]'
  EmptyArrayTypedef ea;
  // CHECK: CXXRecordDecl {{.*}} implicit class __layout_CB_5 definition
  // CHECK-NOT: FieldDecl
}

// check host layout struct with compatible base struct
// CHECK: HLSLBufferDecl {{.*}} line:160:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB {
  // CHECK: VarDecl {{.*}} s8 'hlsl_constant F'
  F s8;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB_6 definition
  // CHECK: FieldDecl {{.*}} s8 '__layout_F'
}
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_F definition
  // CHECK: public 'A'
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_F), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_CB_6), "");

// anonymous structs
// CHECK: HLSLBufferDecl {{.*}} line:175:9 cbuffer CB
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
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
  // CHECK: VarDecl {{.*}} s9 'hlsl_constant struct (unnamed struct at {{.*}}cbuffer.hlsl:177:3
  // CHECK: CXXRecordDecl {{.*}} struct definition
  struct {
    // CHECK: FieldDecl {{.*}} g 'float'
    float g;
    // CHECK: FieldDecl {{.*}} f 'RWBuffer<float>':'hlsl::RWBuffer<float>'
    RWBuffer<float> f;
  } s10;
  // CHECK: VarDecl {{.*}} s10 'hlsl_constant struct (unnamed struct at {{.*}}cbuffer.hlsl:187:3
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_anon definition
  // CHECK: FieldDecl {{.*}} e 'float'
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_anon_1 definition
  // CHECK: FieldDecl {{.*}} g 'float'
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB_7 definition
  // CHECK: FieldDecl {{.*}} s9 '__layout_anon'
  // CHECK: FieldDecl {{.*}} s10 '__layout_anon_1'
}
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_anon), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(OneFloat, __layout_anon_1), "");
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(TwoFloats, __layout_CB_7), "");

// Add uses for the constant buffer declarations so they are not optimized away
export float foo() {
  return a1 + a2 + s1.a + s4.b + s5.c + s8.a + s9.e;
}
