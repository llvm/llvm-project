// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header \
// RUN:   -std=hlsl202x -ast-dump -x hlsl %s | FileCheck %s

// CHECK: VarDecl {{.*}} rm_mat 'float3x3 hlsl_constant __attribute__((row_major))':'matrix<float, 3, 3> hlsl_constant'
row_major float3x3 rm_mat;
// CHECK: VarDecl {{.*}} cm_mat 'float4x4 hlsl_constant __attribute__((column_major))':'matrix<float, 4, 4> hlsl_constant'
column_major float4x4 cm_mat;

// CHECK: CXXRecordDecl {{.*}} struct S definition
struct S {
  // CHECK: FieldDecl {{.*}} m1 'float2x2 __attribute__((row_major))':'matrix<float, 2, 2>'
  row_major float2x2 m1;
  // CHECK: FieldDecl {{.*}} m2 'float3x3 __attribute__((column_major))':'matrix<float, 3, 3>'
  column_major float3x3 m2;
};

// CHECK-LABEL: TypedefDecl {{.*}} RM44 'float4x4 __attribute__((row_major))':'matrix<float, 4, 4>'
// CHECK-NEXT:  AttributedType {{.*}} 'float4x4 __attribute__((row_major))' sugar
typedef row_major float4x4 RM44;

// CHECK-LABEL: TypedefDecl {{.*}} CM44 'float4x4 __attribute__((column_major))':'matrix<float, 4, 4>'
// CHECK-NEXT:  AttributedType {{.*}} 'float4x4 __attribute__((column_major))' sugar
typedef column_major float4x4 CM44;
