// RUN: %clang_cc1 -ast-dump -std=c++26 %s | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -std=c++26 -emit-pch -o %t %s
// RUN: %clang_cc1 %s -std=c++26 -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

template <unsigned N>
void foo() {
  int arr[4] = {1, 2, 3, 4};
  auto [x, ...rest, y] = arr;
  int arr_2 = {rest...};
};

// CHECK-LABEL: FunctionTemplateDecl {{.*}} foo
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: FunctionDecl {{.*}} foo
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: DecompositionDecl {{.*}} 'int[4]'
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int[4]'
// CHECK-NEXT: OpaqueValueExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ArraySubscriptExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: OpaqueValueExpr
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Var {{.*}} 'arr' 'int[4]'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned long'
// CHECK-NEXT: BindingDecl {{.*}} x 'int'
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int[4]' lvalue Decomposition {{.*}} 'int[4]'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: BindingDecl {{.*}} rest 'type-parameter-0-0...'
// CHECK-NEXT: ResolvedUnexpandedPackExpr {{.*}} 'int[4]'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Binding {{.*}} 'rest' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Binding {{.*}} 'rest' 'int'
// CHECK-NEXT: BindingDecl {{.*}} y 'int'
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int[4]' lvalue Decomposition {{.*}} 'int[4]'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} arr_2 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'void'
// CHECK-NEXT: PackExpansionExpr {{.*}} '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'type-parameter-0-0' lvalue Binding {{.*}} 'rest'
#endif
