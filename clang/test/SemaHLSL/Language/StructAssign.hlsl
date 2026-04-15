// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header %s -ast-dump | FileCheck %s

struct Base {
  int A, B;
};

struct Derived: Base {
  float F, G;
};

struct Other {
  int C, D;
};

// CHECK-LABEL: FunctionDecl {{.*}} fn
export void fn() {
  Base B = {1,2};
  Base C = {5,6};
// CHECK: BinaryOperator {{.*}} 'Base' lvalue '='
// CHECK-NEXT: DeclRefExpr {{.*}} 'Base' lvalue Var {{.*}} 'B' 'Base'
// CHECK-NEXT: DeclRefExpr {{.*}} 'Base' lvalue Var {{.*}} 'C' 'Base'
  B = C;

  Other O = {7,8};
// CHECK: BinaryOperator {{.*}} 'Base' lvalue '='
// CHECK-NEXT: DeclRefExpr {{.*}} 'Base' lvalue Var {{.*}} 'C' 'Base'
// CHECK-NEXT: CStyleCastExpr {{.*}} 'Base' <HLSLElementwiseCast>
// CHECK-NEXT: DeclRefExpr {{.*}} 'Other' lvalue Var {{.*}} 'O' 'Other'
  C = (Base)O;

  int2 I2 = {9,10};
// CHECK: BinaryOperator {{.*}} 'Base' lvalue '='
// CHECK-NEXT: DeclRefExpr {{.*}} 'Base' lvalue Var {{.*}} 'C' 'Base'
// CHECK-NEXT: CStyleCastExpr {{.*}} 'Base' <HLSLElementwiseCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int2':'vector<int, 2>' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue Var {{.*}} 'I2' 'int2':'vector<int, 2>'
  C = (Base)I2;

  Derived D = {1,2,3,4};
// CHECK: BinaryOperator {{.*}} 'Base' lvalue '='
// CHECK-NEXT: DeclRefExpr {{.*}} 'Base' lvalue Var {{.*}} 'B' 'Base'
// CHECK-NEXT: CStyleCastExpr {{.*}} 'Base' <HLSLElementwiseCast>
// CHECK-NEXT: DeclRefExpr {{.*}} 'Derived' lvalue Var {{.*}} 'D' 'Derived'
  B = (Base)D;
}
