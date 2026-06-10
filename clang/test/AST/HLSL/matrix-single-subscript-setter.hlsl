// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef int int4x4 __attribute__((matrix_type(4,4)));

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));

export void setMatrix(out float4x4 M, int index, float4 V) {
// CHECK: BinaryOperator{{.*}} 'vector<float, 4>' lvalue matrixcomponent '='
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4 &__restrict'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'index' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'V' 'float4':'vector<float, 4>'
    M[index] = V;
}

export void setMatrixConstIndex(out int4x4 M, int4x4 N ) {
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' lvalue matrixcomponent '='
// CHECK: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4 &__restrict'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'N' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    M[0] = N[3];

// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' lvalue matrixcomponent '='
// CHECK: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4 &__restrict'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'N' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
    M[1] = N[2];

// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' lvalue matrixcomponent '='
// CHECK: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4 &__restrict'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'N' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    M[2] = N[1];

// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' lvalue matrixcomponent '='
// CHECK: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4 &__restrict'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'N' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
    M[3] = N[0];
}
