// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header \
// RUN:   -std=hlsl202x -ast-dump -x hlsl %s | FileCheck %s

// CHECK: VarDecl {{.*}} rm_mat 'hlsl_constant float3x3'
// CHECK-NEXT: HLSLMatrixLayoutAttr {{.*}} row_major
row_major float3x3 rm_mat;

// CHECK: VarDecl {{.*}} cm_mat 'hlsl_constant float4x4'
// CHECK-NEXT: HLSLMatrixLayoutAttr {{.*}} column_major
column_major float4x4 cm_mat;

// CHECK: CXXRecordDecl {{.*}} struct S definition
struct S {
  // CHECK: FieldDecl {{.*}} m1 'float2x2'
  // CHECK-NEXT: HLSLMatrixLayoutAttr {{.*}} row_major
  row_major float2x2 m1;
  // CHECK: FieldDecl {{.*}} m2 'float3x3'
  // CHECK-NEXT: HLSLMatrixLayoutAttr {{.*}} column_major
  column_major float3x3 m2;
};

// CHECK: TypedefDecl {{.*}} RM44 'float4x4'
// CHECK: HLSLMatrixLayoutAttr {{.*}} row_major
typedef row_major float4x4 RM44;
