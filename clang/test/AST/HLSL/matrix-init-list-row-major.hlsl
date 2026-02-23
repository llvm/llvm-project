// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -finclude-default-header -o - %s | FileCheck %s

// This test verifies that matrix initializer lists in HLSL use row-major
// element ordering. The elements in the AST InitListExpr remain in
// row-major order as written in the source code.

// The AST InitListExpr preserves this row-major source order.
// CHECK: VarDecl {{.*}} m2x2 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
export void test_2x2() {
  float2x2 m2x2 = {1, 2, 3, 4};
}

// CHECK: VarDecl {{.*}} m2x3 'float2x3':'matrix<float, 2, 3>' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
export void test_2x3() {
  float2x3 m2x3 = {1, 2, 3, 4, 5, 6};
}

// CHECK: VarDecl {{.*}} m3x2 'float3x2':'matrix<float, 3, 2>' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'float3x2':'matrix<float, 3, 2>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
export void test_3x2() {
  float3x2 m3x2 = {1, 2, 3, 4, 5, 6};
}
