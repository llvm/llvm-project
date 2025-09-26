// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 

typedef float float2x1 __attribute__((matrix_type(2,1)));
typedef float float2x3 __attribute__((matrix_type(2,3)));
typedef float float2x2 __attribute__((matrix_type(2,2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));

[numthreads(1,1,1)]
void ok() {

  // CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:36> col:12 A 'float2x3':'matrix<float, 2, 3>' cinit
  // CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:36> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
  // CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float2x3':'matrix<float, 2, 3>'
  // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
  // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:27> 'int' 2
  // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29> 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:29> 'int' 3
  // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:31> 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:31> 'int' 4
  // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:33> 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:33> 'int' 5
  // CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:35> 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:35> 'int' 6
  float2x3 A = float2x3(1,2,3,4,5,6);

  // CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:32> col:12 B 'float2x1':'matrix<float, 2, 1>' cinit
  // CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:32> 'float2x1':'matrix<float, 2, 1>' functional cast to float2x1 <NoOp>
  // CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:29> 'float2x1':'matrix<float, 2, 1>'
  // CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'float' 1.000000e+00
  // CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:29> 'float' 2.000000e+00
  float2x1 B = float2x1(1.0,2.0);

  // CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:36> col:12 C 'float2x1':'matrix<float, 2, 1>' cinit
  // CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:36> 'float2x1':'matrix<float, 2, 1>' functional cast to float2x1 <NoOp>
  // CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:32> 'float2x1':'matrix<float, 2, 1>'
  // CHECK-NEXT: UnaryOperator 0x{{[0-9a-fA-F]+}} <col:25, col:26> 'float' prefix '-'
  // CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:26> 'float' 1.000000e+00
  // CHECK-NEXT: UnaryOperator 0x{{[0-9a-fA-F]+}} <col:31, col:32> 'float' prefix '-'
  // CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'float' 2.000000e+00
  float2x1 C = float2x1(-1.0f,-2.0f);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:48> col:12 D 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:48> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:47> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:34> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:34> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:41> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:41> 'int' 4
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:44> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:44> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:47> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:47> 'int' 6
  float2x3 D = float2x3(float2(1,2), 3, 4, 5, 6);
          
// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:55> col:12 E 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:55> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:54> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:34> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:34> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:38, col:48> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:38, col:48> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:45, col:47> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:45> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:45> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:47> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:47> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 0
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:38, col:48> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:38, col:48> 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:45, col:47> 'float2':'vector<float, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:45> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:45> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:47> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:47> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:51> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:51> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:54> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:54> 'int' 6
  float2x3 E = float2x3(float2(1,2), float2(3,4), 5, 6);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:46> col:12 F 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:46> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:45> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:38> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:38> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:38> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 2
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float'
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:39> 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:32, col:38> 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:42> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:42> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:45> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:45> 'int' 6
  float2x3 F = float2x3(float4(1,2,3,4), 5, 6);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:48> col:12 G 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:48> 'float2x3':'matrix<float, 2, 3>' functional cast to float2x3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:47> 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:34, col:40> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:40> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:40> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:34, col:40> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:40> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:40> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:34, col:40> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:40> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:40> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: MatrixSubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float' matrixcomponent
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:41> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:34, col:40> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:34> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:34> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:36> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:36> 'int' 2
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:38> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:38> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:40> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:40> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:44> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:44> 'int' 5
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:47> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:47> 'int' 6
  float2x3 G = float2x3(float2x2(1,2,3,4), 5, 6);  


// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:33> col:12 H 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:33> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:32> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float2':'vector<float, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float2':'vector<float, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:30> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:30> 'int' 3
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:32> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:32> 'int' 4
  float2 Vec2 = float2(1.0, 2.0);   
  float2x2 H = float2x2(Vec2,3,4);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:32> col:12 I 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:32> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:31> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:25> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:27> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'j' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:29> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'k' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:31> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:31> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:31> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'l' 'int'
  int i = 1, j = 2, k = 3, l = 4;
  float2x2 I = float2x2(i,j,k,l);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:38> col:12 J 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:38> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:37> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:27> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:27> 'float' lvalue
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:25, col:27> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:25> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25, col:27> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:25, col:27> 'float' lvalue
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:25, col:27> 'float2':'vector<float, 2>' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:25> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:30, col:32> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:30, col:32> 'float' lvalue .a 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:30> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:35, col:37> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:35, col:37> 'float' lvalue .a 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:35> 'struct S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S'
  struct S { float2 f;  float a;} s;
  float2x2 J = float2x2(s.f, s.a, s.a);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:32> col:12 K 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:32> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:31> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:25> '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:25> 'struct T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:27> '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:27> 'struct T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29> 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} <col:29> 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:29> '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29> 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:29> 'struct T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:31> 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} <col:31> 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:31> '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:31> 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:31> 'struct T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T'
  struct T {
    operator float() const { return 1.0f; }
  } t;
  float2x2 K = float2x2(t,t,t,t);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:63> col:28 L 'second_level_of_typedefs':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:32, col:63> 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:41, col:59> 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:41> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:47> 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:53> 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:59> 'float' 4.000000e+00
  typedef float2x2 second_level_of_typedefs;
  second_level_of_typedefs L = float2x2(1.0f, 2.0f, 3.0f, 4.0f);

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:63> col:12 M 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:16, col:63> 'second_level_of_typedefs':'matrix<float, 2, 2>' functional cast to second_level_of_typedefs <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:41, col:59> 'second_level_of_typedefs':'matrix<float, 2, 2>'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:41> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:47> 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:53> 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:59> 'float' 4.000000e+00
  float2x2 M = second_level_of_typedefs(1.0f, 2.0f, 3.0f, 4.0f);
}
