// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

typedef float float2x1 __attribute__((matrix_type(2,1)));
typedef float float2x3 __attribute__((matrix_type(2,3)));
typedef float float2x2 __attribute__((matrix_type(2,2)));
typedef float float4x4 __attribute__((matrix_type(4,4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));

[numthreads(1,1,1)]
void ok() {
// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} A 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 6
  float2x3 A = float2x3(1,2,3,4,5,6);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} B 'float2x1':'matrix<float, 2, 1>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x1':'matrix<float, 2, 1>' functional cast to float2x1 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x1':'matrix<float, 2, 1>'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 2.000000e+00
  float2x1 B = float2x1(1.0,2.0);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} C 'float2x1':'matrix<float, 2, 1>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x1':'matrix<float, 2, 1>' functional cast to float2x1 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x1':'matrix<float, 2, 1>'
// CHECK-NEXT: UnaryOperator 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' prefix '-'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 1.000000e+00
// CHECK-NEXT: UnaryOperator 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' prefix '-'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 2.000000e+00
  float2x1 C = float2x1(-1.0f,-2.0f);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} D 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: ExprWithCleanups 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 6
  float2x3 D = float2x3(float2(1,2), 3, 4, 5, 6);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} E 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: ExprWithCleanups 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 6
  float2x3 E = float2x3(float2(1,2), float2(3,4), 5, 6);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} F 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: ExprWithCleanups 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' xvalue
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 6
  float2x3 F = float2x3(float4(1,2,3,4), 5, 6);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} G 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 5
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 6
float2x3 G = float2x3(float2x2(1,2,3,4), 5, 6);  


// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} H 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' 4
  float2 Vec2 = float2(1.0, 2.0);   
  float2x2 H = float2x2(Vec2,3,4);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} I 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'k' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'j' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'l' 'int'
  int i = 1, j = 2, k = 3, l = 4;
  float2x2 I = float2x2(i,j,k,l);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} J 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue .a 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float' lvalue .a 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
  struct S { float2 f;  float a;} s;
  float2x2 J = float2x2(s.f, s.a, s.a);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} L 'second_level_of_typedefs':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 4.000000e+00
  typedef float2x2 second_level_of_typedefs;
  second_level_of_typedefs L = float2x2(1.0f, 2.0f, 3.0f, 4.0f);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} M 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'second_level_of_typedefs':'matrix<float, 2, 2>' functional cast to second_level_of_typedefs <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'second_level_of_typedefs':'matrix<float, 2, 2>'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' 4.000000e+00
  float2x2 M = second_level_of_typedefs(1.0f, 2.0f, 3.0f, 4.0f);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} N 'float4x4':'matrix<float, 4, 4>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float4x4':'matrix<float, 4, 4>' functional cast to float4x4 <HLSLElementwiseCast>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'sF' lvalue Var 0x{{[0-9a-fA-F]+}} 'f' 'sF'
struct sF {
    float f[16];
};

sF f;
float4x4 N = float4x4(f);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} GettingStrange 'float2x1':'matrix<float, 2, 1>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x1':'matrix<float, 2, 1>' functional cast to float2x1 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x1':'matrix<float, 2, 1>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S2' lvalue Var 0x{{[0-9a-fA-F]+}} 's2' 'S2'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S2' lvalue Var 0x{{[0-9a-fA-F]+}} 's2' 'S2'
struct S2 { float f; };
S2 s2;
float2x1 GettingStrange = float2x1(s2, s2);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} GettingStrange2 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S3' lvalue Var 0x{{[0-9a-fA-F]+}} 's3' 'S3'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S3' lvalue Var 0x{{[0-9a-fA-F]+}} 's3' 'S3'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S3' lvalue Var 0x{{[0-9a-fA-F]+}} 's3' 'S3'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S3' lvalue Var 0x{{[0-9a-fA-F]+}} 's3' 'S3'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> '__size_t':'unsigned long' 1
struct S3 { float2 f;};
S3 s3;
float2x2 GettingStrange2 = float2x2(s3, s3);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> col:{{[0-9]+}} GettingStrange3 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <HLSLElementwiseCast>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:{{[0-9]+}}> 'S4' lvalue Var 0x{{[0-9a-fA-F]+}} 's4' 'S4'
struct S4 { float4 f;};
S4 s4;
float2x2 GettingStrange3 = float2x2(s4);

}

