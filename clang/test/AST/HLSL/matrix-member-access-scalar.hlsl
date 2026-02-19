// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float3x3 __attribute__((matrix_type(3,3)));

[numthreads(1,1,1)]
void ok() {
    float3x3 A;

   // CHECK:      BinaryOperator {{.*}} 'vector<float, 1>' lvalue '='
   // CHECK-NEXT: MatrixElementExpr {{.*}} 'vector<float, 1>' lvalue _m12
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
   // CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.140000e+00
    A._m12 = 3.14;

   // CHECK: VarDecl {{.*}} r 'float' cinit
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <HLSLVectorTruncation>
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <LValueToRValue>
   // CHECK-NEXT: MatrixElementExpr {{.*}}  'vector<float, 1>' lvalue _m00
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
    float r = A._m00;

   // CHECK: VarDecl {{.*}} good1 'float' cinit
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <HLSLVectorTruncation>
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <LValueToRValue>
   // CHECK-NEXT: MatrixElementExpr {{.*}}  'vector<float, 1>' lvalue _11
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
    float good1 = A._11;

   // CHECK:      BinaryOperator {{.*}} 'vector<float, 1>' lvalue '='
   // CHECK-NEXT: MatrixElementExpr {{.*}} 'vector<float, 1>' lvalue _33
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float3x3':'matrix<float, 3, 3>' lvalue Var {{.*}} 'A' 'float3x3':'matrix<float, 3, 3>'
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' <VectorSplat>
   // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
   // CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue Var {{.*}} 'R' 'float'
    float R;
    A._33 = R;
}
