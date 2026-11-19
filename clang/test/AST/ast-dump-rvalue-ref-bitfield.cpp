// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -fsyntax-only \
// RUN:   -ast-dump -ast-dump-filter test %s | FileCheck %s

// static_cast<T&&> applied to a bit-field lvalue. A bit-field has no address,
// so the reference binds to a materialized temporary initialized from an
// LValueToRValue conversion of the bit-field.

struct A {
  int a : 4;
};
// CHECK-LABEL: FunctionDecl {{.*}} test 'void (A)'
void test(A obj) {
// CHECK:       VarDecl {{.*}} r 'int &&' cinit
// CHECK-NEXT:  `-ExprWithCleanups {{.*}} 'int' xvalue
// CHECK-NEXT:    `-CXXStaticCastExpr {{.*}} 'int' xvalue static_cast<int &&> <NoOp>
// CHECK-NEXT:      `-MaterializeTemporaryExpr {{.*}} 'int' xvalue extended by Var {{.*}} 'r' 'int &&'
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:          `-MemberExpr {{.*}} 'int' lvalue bitfield .a
// CHECK-NEXT:            `-DeclRefExpr {{.*}} 'A' lvalue ParmVar {{.*}} 'obj' 'A'
  int &&r = static_cast<int &&>(obj.a);
  
// CHECK:       VarDecl {{.*}} cr 'const int &&' cinit
// CHECK-NEXT:  `-ExprWithCleanups {{.*}} 'const int' xvalue
// CHECK-NEXT:    `-CXXStaticCastExpr {{.*}} 'const int' xvalue static_cast<const int &&> <NoOp>
// CHECK-NEXT:      `-MaterializeTemporaryExpr {{.*}} 'const int' xvalue extended by Var {{.*}} 'cr' 'const int &&'
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'const int' <NoOp>
// CHECK-NEXT:          `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:            `-MemberExpr {{.*}} 'int' lvalue bitfield .a
// CHECK-NEXT:              `-DeclRefExpr {{.*}} 'A' lvalue ParmVar {{.*}} 'obj' 'A'
  const int &&cr = static_cast<const int &&>(obj.a);
}
