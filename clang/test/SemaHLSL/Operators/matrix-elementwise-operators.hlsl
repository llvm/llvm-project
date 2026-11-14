// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -ast-dump -ast-dump-filter=test -verify | FileCheck %s

using namespace hlsl;

// CHECK-LABEL: FunctionDecl {{.*}} test_mul 'float2x3 (float2x3, float2x3)'
// CHECK: BinaryOperator {{.*}} 'float2x3':'matrix<float, 2, 3>' '*'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x3':'matrix<float, 2, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x3':'matrix<float, 2, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'float2x3':'matrix<float, 2, 3>'
float2x3 test_mul(float2x3 a, float2x3 b) {
  return a * b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_div 'float2x3 (float2x3, float2x3)'
// CHECK: BinaryOperator {{.*}} 'float2x3':'matrix<float, 2, 3>' '/'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x3':'matrix<float, 2, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'float2x3':'matrix<float, 2, 3>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x3':'matrix<float, 2, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'float2x3':'matrix<float, 2, 3>'
float2x3 test_div(float2x3 a, float2x3 b) {
  return a / b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_neg 'float2x3 (float2x3)'
// CHECK: UnaryOperator {{.*}} 'float2x3':'matrix<float, 2, 3>' prefix '-'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x3':'matrix<float, 2, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'float2x3':'matrix<float, 2, 3>'
float2x3 test_neg(float2x3 a) {
  return -a;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_plus 'float2x3 (float2x3)'
// CHECK: UnaryOperator {{.*}} 'float2x3':'matrix<float, 2, 3>' prefix '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float2x3':'matrix<float, 2, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'float2x3':'matrix<float, 2, 3>'
float2x3 test_plus(float2x3 a) {
  return +a;
}

// `*` is element-wise, so the operands must have the same dimensions.
float2x3 test_mul_mismatch(float2x3 a, float3x2 b) {
  return a * b; // expected-error {{invalid operands to binary expression ('float2x3' (aka 'matrix<float, 2, 3>') and 'float3x2' (aka 'matrix<float, 3, 2>'))}}
}
