// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -ast-dump -o - %s | FileCheck %s

// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 4]]:9 cbuffer CB
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK-NEXT: HLSLResourceBindingAttr {{.*}} "b3" "space2"
// CHECK-NEXT: VarDecl {{.*}} used a 'hlsl_constant float'
cbuffer CB : register(b3, space2) {
  float a;
}

// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 4]]:9 tbuffer TB
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit SRV
// CHECK-NEXT: HLSLResourceBindingAttr {{.*}} "t2" "space1"
// CHECK-NEXT: VarDecl {{.*}} used b 'hlsl_constant float'
tbuffer TB : register(t2, space1) {
  float b;
}

export float foo() {
  return a + b;
}

// CHECK: VarDecl {{.*}} UAV 'RWBuffer<float>':'hlsl::RWBuffer<float>'
// CHECK: HLSLResourceBindingAttr {{.*}} "u3" "space0"
RWBuffer<float> UAV : register(u3);

// CHECK: VarDecl {{.*}} UAV1 'RWBuffer<float>':'hlsl::RWBuffer<float>'
// CHECK: HLSLResourceBindingAttr {{.*}} "u2" "space0"
// CHECK: VarDecl {{.*}} UAV2 'RWBuffer<float>':'hlsl::RWBuffer<float>'
// CHECK: HLSLResourceBindingAttr {{.*}} "u4" "space0"
RWBuffer<float> UAV1 : register(u2), UAV2 : register(u4);

//
// Default constants ($Globals) layout annotations

// CHECK: VarDecl {{.*}} f 'hlsl_constant float'
// CHECK: HLSLResourceBindingAttr {{.*}} "c5" "space0"
float f : register(c5);

// CHECK: VarDecl {{.*}} intv 'hlsl_constant int4':'vector<int hlsl_constant, 4>'
// CHECK: HLSLResourceBindingAttr {{.*}} "c2" "space0"
int4 intv : register(c2);

// CHECK: VarDecl {{.*}} dar 'hlsl_constant double[5]'
// CHECK: HLSLResourceBindingAttr {{.*}} "c3" "space0"
double dar[5] :  register(c3);

struct S {
  int a;
};

// CHECK: VarDecl {{.*}} s 'hlsl_constant S'
// CHECK: HLSLResourceBindingAttr {{.*}} "c10" "space0
S s : register(c10);
