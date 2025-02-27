// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.3-library -S -finclude-default-header -fnative-half-type -ast-dump  -x hlsl %s | FileCheck %s


// CHECK: HLSLBufferDecl {{.*}} cbuffer A
cbuffer A
{
    // CHECK-NEXT:-HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
    // CHECK-NEXT:-HLSLResourceAttr {{.*}} <<invalid sloc>> Implicit CBuffer
    // CHECK-NEXT: VarDecl {{.*}} A1 'hlsl_constant float4'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 0
    float4 A1 : packoffset(c);
    // CHECK-NEXT: VarDecl {{.*}} col:11 A2 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 1 0
    float A2 : packoffset(c1);
    // CHECK-NEXT: VarDecl {{.*}} col:11 A3 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 1 1
    float A3 : packoffset(c1.y);
}

// CHECK: HLSLBufferDecl {{.*}} cbuffer B
cbuffer B
{
    // CHECK: VarDecl {{.*}} B0 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 1
    float B0 : packoffset(c0.g);
    // CHECK-NEXT: VarDecl {{.*}} B1 'hlsl_constant double'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 2
	double B1 : packoffset(c0.b);
    // CHECK-NEXT: VarDecl {{.*}} B2 'hlsl_constant half'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 0
	half B2 : packoffset(c0.r);
}

// CHECK: HLSLBufferDecl {{.*}} cbuffer C
cbuffer C
{
    // CHECK: VarDecl {{.*}} C0 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 1
    float C0 : packoffset(c0.y);
    // CHECK-NEXT: VarDecl {{.*}} C1 'hlsl_constant float2'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 2
	float2 C1 : packoffset(c0.z);
    // CHECK-NEXT: VarDecl {{.*}} C2 'hlsl_constant half'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0
	half C2 : packoffset(c0.x);
}


// CHECK: HLSLBufferDecl {{.*}} cbuffer D
cbuffer D
{
    // CHECK: VarDecl {{.*}} D0 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 1
    float D0 : packoffset(c0.y);
    // CHECK-NEXT: VarDecl {{.*}} D1 'hlsl_constant float[2]'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 1 0
	float D1[2] : packoffset(c1.x);
    // CHECK-NEXT: VarDecl {{.*}} D2 'hlsl_constant half3'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 2 1
	half3 D2 : packoffset(c2.y);
    // CHECK-NEXT: VarDecl {{.*}} D3 'hlsl_constant double'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 2
	double D3 : packoffset(c0.z);
}

struct ST {
  float a;
  float2 b;
  half c;
};

// CHECK: HLSLBufferDecl {{.*}} cbuffer S
cbuffer S {
    // CHECK: VarDecl {{.*}} S0 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 1
  float S0 : packoffset(c0.y);
    // CHECK: VarDecl {{.*}} S1 'hlsl_constant ST'
    // CHECK: HLSLPackOffsetAttr {{.*}} 1 0
  ST S1 : packoffset(c1);
    // CHECK: VarDecl {{.*}} S2 'hlsl_constant double2'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 2 0
  double2 S2 : packoffset(c2);
}

struct ST2 {
  float s0;
  ST s1;
  half s2;
};

// CHECK: HLSLBufferDecl {{.*}} cbuffer S2
cbuffer S2 {
    // CHECK: VarDecl {{.*}} S20 'hlsl_constant float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0 3
  float S20 : packoffset(c0.a);
    // CHECK: VarDecl {{.*}} S21 'hlsl_constant ST2'
    // CHECK: HLSLPackOffsetAttr {{.*}} 1 0
  ST2 S21 : packoffset(c1);
    // CHECK: VarDecl {{.*}} S22 'hlsl_constant half'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 3 1
  half S22 : packoffset(c3.y);
}
