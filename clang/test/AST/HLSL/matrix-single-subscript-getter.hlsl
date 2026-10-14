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
    return M[0] + M[1] + M[2] + M[3];
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
   return M[0] + M[1] + M[2] + M[3];
}

export vector<bool, 3> getBoolVecFromTemplateMat(matrix<bool, 2, 3> M) {
    // CHECK: FunctionDecl {{.*}} used getBoolVecFromTemplateMat 'vector<bool, 3> (matrix<bool, 2, 3>)'
    // CHECK-NEXT: ParmVarDecl {{.*}} used M 'matrix<bool, 2, 3>'
    // CHECK-NEXT: CompoundStmt {{.*}}
    // CHECK-NEXT: ReturnStmt {{.*}}
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 3>' <LValueToRValue>
    // CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<bool, 3>' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr {{.*}} 'matrix<bool, 2, 3>' lvalue ParmVar {{.*}} 'M' 'matrix<bool, 2, 3>'
    // CHECK-NEXT: IntegerLiteral {{.*}}'int' 0
    return M[0];
}

template<typename T>
vector<T, 3> getVecFromTemplateMat(matrix<T, 2, 3> M) {
    // CHECK: FunctionTemplateDecl {{.*}} getVecFromTemplateMat
    // CHECK-NEXT: TemplateTypeParmDecl {{.*}}  referenced typename depth 0 index 0 T
    // CHECK-NEXT: FunctionDecl {{.*}} getVecFromTemplateMat 'vector<T, 3> (matrix<T, 2, 3>)'
    // CHECK-NEXT: ParmVarDecl {{.*}} referenced M 'matrix<T, 2, 3>'
    // CHECK-NEXT: CompoundStmt {{.*}}
    // CHECK-NEXT: ReturnStmt {{.*}}
    // CHECK-NEXT: MatrixSubscriptExpr {{.*}} '<incomplete matrix index type>' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr {{.*}} 'matrix<T, 2, 3>' lvalue ParmVar {{.*}} 'M' 'matrix<T, 2, 3>'
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: FunctionDecl {{.*}} used getVecFromTemplateMat 'vector<bool, 3> (matrix<bool, 2, 3>)' implicit_instantiation instantiated_from {{.*}}
    // CHECK-NEXT: TemplateArgument type 'bool'
    // CHECK-NEXT: BuiltinType {{.*}} 'bool'
    // CHECK-NEXT: ParmVarDecl {{.*}} used M 'matrix<bool, 2, 3>'
    // CHECK-NEXT: CompoundStmt {{.*}}
    // CHECK-NEXT: ReturnStmt {{.*}}
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 3>' <LValueToRValue>
    // CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<bool, 3>' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr {{.*}} <col:12> 'matrix<bool, 2, 3>' lvalue ParmVar {{.*}} 'M' 'matrix<bool, 2, 3>'
    // CHECK-NEXT: IntegerLiteral {{.*}} <col:14> 'int' 0
    // CHECK-NEXT: TypedefDecl {{.*}} referenced bool3 'vector<bool, 3>'
    return M[0];
}

typedef bool bool3 __attribute__((ext_vector_type(3)));
typedef bool bool2x3 __attribute__((matrix_type(2,3)));

export bool3 testTemplatedMatrixAccess(bool2x3 M) {
  return getVecFromTemplateMat(M);
}
