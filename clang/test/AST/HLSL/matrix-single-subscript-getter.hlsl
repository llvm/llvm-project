// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef int int4x4 __attribute__((matrix_type(4,4)));

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));

export float4 getFloatMatrixDynamic(float4x4 M, int index) {
// CHECK: FunctionDecl {{.*}} used getFloatMatrixDynamic 'float4 (float4x4, int)'
// CHECK: ReturnStmt {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4':'matrix<float, 4, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'index' 'int'
    return M[index];
}

export int4 getIntMatrixDynamic(int4x4 M, int index) {
// CHECK: FunctionDecl {{.*}} used getIntMatrixDynamic 'int4 (int4x4, int)'
// CHECK: ReturnStmt {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'index' 'int'
    return M[index];
}

export float4 AddFloatMatrixConstant(float4x4 M) {
// CHECK: FunctionDecl {{.*}} used AddFloatMatrixConstant 'float4 (float4x4)'
// CHECK: ReturnStmt {{.*}}
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<float, 4>' '+'
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<float, 4>' '+'
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<float, 4>' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4':'matrix<float, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4':'matrix<float, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4':'matrix<float, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' lvalue ParmVar {{.*}} 'M' 'float4x4':'matrix<float, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    return M[0] + M[1] + M[2] +M[3];
}

export int4 AddIntMatrixConstant(int4x4 M) {
// CHECK: FunctionDecl {{.*}} used AddIntMatrixConstant 'int4 (int4x4)'
// CHECK: ReturnStmt {{.*}}
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<int, 4>' '+'
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<int, 4>' '+'
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<int, 4>' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<int, 4>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' lvalue ParmVar {{.*}} 'M' 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
   return M[0] + M[1] + M[2] +M[3];
}