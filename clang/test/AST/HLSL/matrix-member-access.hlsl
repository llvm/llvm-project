// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float3x3 __attribute__((matrix_type(3,3)));

[numthreads(1,1,1)]
void ok() {
    float3x3 A;
    // CHECK: BinaryOperator 0x{{[0-9a-fA-F]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue matrixcomponent '='
    // CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float3x3':'matrix<float, 3, 3>' lvalue Var 0x{{[0-9a-fA-F]+}} 'A' 'float3x3':'matrix<float, 3, 3>'
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 1
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 2
    // CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 3.140000e+00
    A._m12 = 3.14;

    // CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} r 'float' cinit
    // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
    // CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float3x3':'matrix<float, 3, 3>' lvalue Var 0x{{[0-9a-fA-F]+}} 'A' 'float3x3':'matrix<float, 3, 3>'
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 0
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 0
    float r = A._m00;

    // CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} good1 'float' cinit
    // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
    // CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float3x3':'matrix<float, 3, 3>' lvalue Var 0x{{[0-9a-fA-F]+}} 'A' 'float3x3':'matrix<float, 3, 3>'
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 0
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 0
    float good1 = A._11;

    // CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} good2 'float' cinit
    // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
    // CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue matrixcomponent
    // CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float3x3':'matrix<float, 3, 3>' lvalue Var 0x{{[0-9a-fA-F]+}} 'A' 'float3x3':'matrix<float, 3, 3>'
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 2
    // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'unsigned int' 2    
    float good2 = A._33;
}
