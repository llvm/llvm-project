// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -finclude-default-header -o - %s | FileCheck %s

// This test verifies that matrix initializer lists in HLSL use row-major
// element ordering. The elements in the AST InitListExpr remain in
// row-major order as written in the source code.

// The AST InitListExpr preserves this row-major source order.
// CHECK: VarDecl {{.*}} m2x2 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 4.000000e+00
export void test_2x2() {
  float2x2 m2x2 = {1.0, 2.0, 3.0, 4.0};
}

// CHECK: VarDecl {{.*}} m2x3 'int2x3':'matrix<int, 2, 3>' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'int2x3':'matrix<int, 2, 3>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
export void test_2x3() {
  int2x3 m2x3 = {1, 2, 3, 4, 5, 6};
}

// CHECK: VarDecl {{.*}} m3x2 'bool3x2':'matrix<bool, 3, 2>' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'bool3x2':'matrix<bool, 3, 2>'
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true
export void test_3x2() {
  bool3x2 m3x2 = {true, false, false, true, true, true};
}
