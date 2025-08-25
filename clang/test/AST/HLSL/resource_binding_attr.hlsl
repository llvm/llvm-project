// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=CHECK,SPV

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

// CHECK: VarDecl {{.*}} UAV3 'RWBuffer<float>':'hlsl::RWBuffer<float>'
// CHECK: HLSLResourceBindingAttr {{.*}} "" "space5"
RWBuffer<float> UAV3 : register(space5);

// CHECK: VarDecl {{.*}} UAV_Array 'RWBuffer<float>[10]'
// CHECK: HLSLResourceBindingAttr {{.*}} "u10" "space6"
RWBuffer<float> UAV_Array[10] : register(u10, space6);

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

//
// Implicit binding

// Constant buffers should have implicit binding attribute added by SemaHLSL,
// unless the target is SPIR-V and there is [[vk::binding]] attribute.
// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 3]]:9 cbuffer CB2
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK-NEXT: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
cbuffer CB2 {
  float4 c;
}

// CHECK: HLSLBufferDecl {{.*}} line:[[# @LINE + 7]]:9 cbuffer CB3
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit
// DXIL-NOT: HLSLVkBindingAttr
// SPV: HLSLVkBindingAttr {{.*}} 1 0
// SPV-NOT: HLSLResourceBindingAttr {{.*}} Implicit
[[vk::binding(1)]]
cbuffer CB3 {
  float2 d;
}

// Resource arrays should have implicit binding attribute added by SemaHLSL,
// unless the target is SPIR-V and there is [[vk::binding]] attribute.
// CHECK: VarDecl {{.*}} SB 'StructuredBuffer<float>[10]'
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
StructuredBuffer<float> SB[10];

// CHECK: VarDecl {{.*}} SB2 'StructuredBuffer<float>[10]'
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit
// DXIL-NOT: HLSLVkBindingAttr
// SPV: HLSLVkBindingAttr {{.*}} 2 0
// SPV-NOT: HLSLResourceBindingAttr {{.*}} Implicit
[[vk::binding(2)]]
StructuredBuffer<float> SB2[10];

// $Globals should have implicit binding attribute added by SemaHLSL
// CHECK: HLSLBufferDecl {{.*}} implicit cbuffer $Globals
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
