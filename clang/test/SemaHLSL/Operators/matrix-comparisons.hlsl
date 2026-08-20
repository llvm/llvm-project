// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -Wno-implicit-int-float-conversion %s -ast-dump -ast-dump-filter=test | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -Wno-implicit-int-float-conversion %s -DERRORS -verify

// CHECK-LABEL: FunctionDecl {{.*}} test_matrix_matrix 'bool2x2 (float2x2, float2x2)'
// CHECK: BinaryOperator {{.*}} 'matrix<bool, 2, 2>' '<'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' lvalue ParmVar {{.*}} 'a'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' lvalue ParmVar {{.*}} 'b'
bool2x2 test_matrix_matrix(float2x2 a, float2x2 b) {
  return a < b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_matrix_scalar 'bool2x2 (float2x2, int)'
// CHECK: BinaryOperator {{.*}} 'matrix<bool, 2, 2>' '=='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' lvalue ParmVar {{.*}} 'a'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'b'
bool2x2 test_matrix_scalar(float2x2 a, int b) {
  return a == b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_scalar_matrix 'bool2x2 (int, float2x2)'
// CHECK: BinaryOperator {{.*}} 'matrix<bool, 2, 2>' '>='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'a'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' lvalue ParmVar {{.*}} 'b'
bool2x2 test_scalar_matrix(int a, float2x2 b) {
  return a >= b;
}

#ifdef ERRORS

bool2x2 test_dimension_mismatch(float2x2 a, float3x3 b) {
  return a != b; // expected-error {{invalid operands to binary expression ('float2x2' (aka 'matrix<float, 2, 2>') and 'float3x3' (aka 'matrix<float, 3, 3>'))}}
}

bool2x2 test_element_mismatch(float2x2 a, int2x2 b) {
  return a < b; // expected-error {{invalid operands to binary expression ('float2x2' (aka 'matrix<float, 2, 2>') and 'int2x2' (aka 'matrix<int, 2, 2>'))}}
}

struct Unsupported {};

bool2x2 test_unsupported_operand(float2x2 a, Unsupported b) {
  return a > b; // expected-error {{invalid operands to binary expression ('float2x2' (aka 'matrix<float, 2, 2>') and 'Unsupported')}} expected-error {{cannot initialize a value of type 'float' with an rvalue of type 'Unsupported'}}
}

#endif
