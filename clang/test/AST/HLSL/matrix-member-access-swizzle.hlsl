// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float3x3 __attribute__((matrix_type(3,3)));
typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

[numthreads(1,1,1)]
void ok() {
    float3x3 A;

   // CHECK:      BinaryOperator {{.*}} 'float2':'vector<float, 2>' lvalue '='
   // CHECK-NEXT: MatrixElementExpr {{.*}} 'float2':'vector<float, 2>' lvalue _m12_m21
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
   // CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float2':'vector<float, 2>' xx
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
   // CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.140000e+00
    A._m12_m21 = 3.14.xx;

   // CHECK: VarDecl {{.*}}r 'float2':'vector<float, 2>' cinit
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2':'vector<float, 2>' <LValueToRValue>
   // CHECK-NEXT: MatrixElementExpr {{.*}}  'float2':'vector<float, 2>' lvalue _m00_m11
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
    float2 r = A._m00_m11;

   // CHECK: VarDecl {{.*}} good1 'float3':'vector<float, 3>' cinit
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float3':'vector<float, 3>' <LValueToRValue>
   // CHECK-NEXT: MatrixElementExpr {{.*}}  'float3':'vector<float, 3>' lvalue _11_22_33
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
    float3 good1 = A._11_22_33;

   // CHECK:      BinaryOperator {{.*}} 'float4':'vector<float, 4>' lvalue '='
   // CHECK-NEXT: MatrixElementExpr {{.*}} 'float4':'vector<float, 4>' lvalue _11_22_33_44
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue Var {{.*}} 'B' 'float4x4':'matrix<float, 4, 4>'
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue Var {{.*}} 'R' 'float4':'vector<float, 4>'
    float4 R;
    float4x4 B;
    B._11_22_33_44 = R;

    // CHECK: BinaryOperator {{.*}} 'float3':'vector<float, 3>' lvalue '='
    // CHECK-NEXT: MatrixElementExpr {{.*}} 'float3':'vector<float, 3>' lvalue _11_22_33
    // CHECK-NEXT: DeclRefExpr{{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
    // CHECK-NEXT: ImplicitCastExpr {{.*}}'float3':'vector<float, 3>' <LValueToRValue>
    // CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float3':'vector<float, 3>' lvalue vectorcomponent rgb
    // CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue Var {{.*}} 'R' 'float4':'vector<float, 4>'
    A._11_22_33 = R.rgb;
}
