// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float4x2 __attribute__((matrix_type(4,2)));
typedef float float2x2 __attribute__((matrix_type(2,2)));
typedef int int4x4 __attribute__((matrix_type(4,4)));


[numthreads(1,1,1)]
void ok() {

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} used m 'float4x2':'matrix<float, 4, 2>' cinit
// CHECK-NEXT: ExprWithCleanups 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 3>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 3>' xxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 3>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 3>' xxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 2>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 2>' xx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 2>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 2>' xx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 3>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 3>' xxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'vector<int, 2>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'vector<int, 2>' xx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'int' x
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'vector<int, 2>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}, col:{{[0-9a-fA-F]+}}> 'vector<int, 2>' xx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9a-fA-F]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
float4x2 m = {1.xxx, 2.xx, 3.x, 4.xx};

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} s 'S' cinit
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'S'
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float4x2':'matrix<float, 4, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm' 'float4x2':'matrix<float, 4, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
struct S { float2x2 x; float2x2 y;};
S s = {m};

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} used m2 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: ExprWithCleanups 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'vector<int, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'vector<int, 1>' <VectorSplat>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 3
float2x2 m2 = {0.xxxx};

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} m3 'int4x4':'matrix<int, 4, 4>' cinit
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'int4x4':'matrix<int, 4, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'm2' 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
int4x4 m3 = {m2, m2, m2, m2};

}