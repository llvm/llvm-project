
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef int int4x4 __attribute__((matrix_type(4,4)));

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));

export void setMatrix(out float4x4 M, int index, float4 V) {
// CHECK: FunctionDecl {{.*}} used setMatrix 'void (out float4x4, int, float4)'
// CHECK: BinaryOperator {{.*}} 'float4':'vector<float, 4>' lvalue vectorcomponent '='
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float4':'vector<float, 4>' lvalue vectorcomponent abgr
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4 &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'index' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'V' 'float4':'vector<float, 4>'
    M[index].abgr = V;
}

export void setMatrix1(out float4x4 M, float4 V) {
// CHECK: FunctionDecl {{.*}} used setMatrix1 'void (out float4x4, float4)'
// CHECK: BinaryOperator {{.*}} 'float4':'vector<float, 4>' lvalue vectorcomponent '='
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float4':'vector<float, 4>' lvalue vectorcomponent abgr
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4 &__restrict'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'V' 'float4':'vector<float, 4>'
    M[3].abgr = V;
}

export void setMatrix2(out int4x4 M, int4 V) {
// CHECK: FunctionDecl {{.*}} used setMatrix2 'void (out int4x4, int4)'
// CHECK: BinaryOperator {{.*}} 'int4':'vector<int, 4>' lvalue vectorcomponent '='
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'int4':'vector<int, 4>' lvalue vectorcomponent rgba
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4 &__restrict'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int4':'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4':'vector<int, 4>' lvalue ParmVar {{.*}} 'V' 'int4':'vector<int, 4>'
    M[2].rgba = V;
}

export float3 getMatrix(float4x4 M, int index) {
// CHECK: FunctionDecl {{.*}} used getMatrix 'float3 (float4x4, int)'
// CHECK: ReturnStmt {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float3':'vector<float, 3>' <LValueToRValue>
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'float3':'vector<float, 3>' lvalue vectorcomponent rgb
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4':'matrix<float, 4, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'index' 'int'
    return M[index].rgb;
}
