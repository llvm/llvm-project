// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump -o - %s | FileCheck %s

struct EmptyStruct {
};

struct S {
  RWBuffer<float> buf;
  EmptyStruct es;
  float ea[0];
  float b;
};

// CHECK: VarDecl {{.*}} used a 'hlsl_constant float'
float a;

// CHECK: VarDecl {{.*}} b 'RWBuffer<float>':'hlsl::RWBuffer<float>'
RWBuffer<float> b; 

// CHECK: VarDecl {{.*}} c 'EmptyStruct'
EmptyStruct c;

// CHECK: VarDecl {{.*}} d 'float[0]'
float d[0];

// CHECK: VarDecl {{.*}} e 'RWBuffer<float>[2]'
RWBuffer<float> e[2];

// CHECK: VarDecl {{.*}} f 'groupshared float'
groupshared float f;

// CHECK: VarDecl {{.*}} g 'hlsl_constant float'
float g;

// CHECK: VarDecl {{.*}} h 'hlsl_constant S'
S h;

// CHECK: HLSLBufferDecl {{.*}} implicit cbuffer $Globals
// CHECK: CXXRecordDecl {{.*}} implicit struct __cblayout_$Globals definition
// CHECK: PackedAttr
// CHECK-NEXT: FieldDecl {{.*}} a 'float'
// CHECK-NEXT: FieldDecl {{.*}} g 'float'
// CHECK-NEXT: FieldDecl {{.*}} h '__cblayout_S'

// CHECK: CXXRecordDecl {{.*}} implicit struct __cblayout_S definition
// CHECK: PackedAttr {{.*}} Implicit
// CHECK-NEXT: FieldDecl {{.*}} b 'float'

export float foo() {
  return a;
}
