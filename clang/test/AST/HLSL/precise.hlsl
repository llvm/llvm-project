// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -std=hlsl202x -finclude-default-header -ast-dump -verify %s | FileCheck %s
// expected-no-diagnostics

// CHECK-LABEL: TranslationUnitDecl
// CHECK-NEXT: HLSLPreciseAttr {{.*}} Implicit

// CHECK-LABEL: TypedefDecl {{.*}} pfloat 'float'
// CHECK-NEXT: BuiltinType {{.*}} 'float'
// CHECK-NEXT: HLSLPreciseAttr
typedef precise float pfloat;

// CHECK-LABEL: VarDecl {{.*}} typed_global 'hlsl_constant pfloat':'hlsl_constant float'
pfloat typed_global;

// CHECK-LABEL: VarDecl {{.*}} global 'hlsl_constant float'
// CHECK-NEXT: HLSLPreciseAttr
precise float global;

// CHECK-LABEL: VarDecl {{.*}} after_type 'hlsl_constant float'
// CHECK-NEXT: HLSLPreciseAttr
float precise after_type;

// CHECK-LABEL: VarDecl {{.*}} first 'hlsl_constant float'
// CHECK-NEXT: HLSLPreciseAttr
// CHECK-LABEL: VarDecl {{.*}} second 'hlsl_constant float'
// CHECK-NEXT: HLSLPreciseAttr
precise float first, second;

struct S {
  // CHECK-LABEL: FieldDecl {{.*}} scalar 'float'
  // CHECK-NEXT: HLSLPreciseAttr
  precise float scalar;

  // CHECK-LABEL: FieldDecl {{.*}} vec 'float4'
  // CHECK-NEXT: HLSLPreciseAttr
  precise float4 vec;

  // CHECK-LABEL: FieldDecl {{.*}} mat 'float2x2'
  // CHECK-NEXT: HLSLPreciseAttr
  precise float2x2 mat;

  // CHECK-LABEL: FieldDecl {{.*}} array 'float[2]'
  // CHECK-NEXT: HLSLPreciseAttr
  precise float array[2];
};

// CHECK-LABEL: FunctionDecl {{.*}} function 'float (float)'
// CHECK-NEXT: ParmVarDecl {{.*}} input 'float'
// CHECK-NEXT: HLSLPreciseAttr
// CHECK-NEXT: HLSLPreciseAttr
precise float function(precise float input);

// CHECK-LABEL: FunctionDecl {{.*}} parameters 'void (
// CHECK: ParmVarDecl {{.*}} input 'float'
// CHECK-NEXT: HLSLPreciseAttr
// CHECK: ParmVarDecl {{.*}} output
// CHECK-NEXT: HLSLPreciseAttr
// CHECK: ParmVarDecl {{.*}} both
// CHECK: HLSLPreciseAttr
void parameters(precise in float input, precise out float output,
                inout precise float both);

void locals() {
  // CHECK-LABEL: VarDecl {{.*}} local 'float'
  // CHECK-NEXT: HLSLPreciseAttr
  precise float local;

  // CHECK-LABEL: VarDecl {{.*}} constant 'const float'
  // CHECK: HLSLPreciseAttr
  const precise float constant = 1.0;

  // CHECK-LABEL: VarDecl {{.*}} initialized 'float'
  // CHECK: HLSLPreciseAttr
  precise float initialized(float(1.0));

  // CHECK-LABEL: VarDecl {{.*}} index 'float'
  // CHECK: HLSLPreciseAttr
  for (precise float index = 0.0; index < 2.0; index += 1.0) {
  }
}
