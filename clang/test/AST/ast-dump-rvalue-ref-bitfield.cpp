// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++11 -fsyntax-only \
// RUN:   -ast-dump -ast-dump-filter test %s | FileCheck %s

// static_cast<T&&> applied to a bit-field lvalue. A bit-field has no address,
// so the reference has to bind to a materialized temporary.
//
// FIXME: Clang currently binds the reference directly to the bit-field. No
// MaterializeTemporaryExpr or LValueToRValue conversion is created, and this
// AST crashes CodeGen (see CodeGenCXX/rvalue-ref-bitfield.cpp).

struct A {
  int a : 4;
  int b : 1;
};

void test(A obj) {
  int &&r = static_cast<int &&>(obj.a);
  const int &&cr = static_cast<const int &&>(obj.a);
}

// CHECK-LABEL: FunctionDecl {{.*}} test 'void (A)'
// CHECK:       VarDecl {{.*}} r 'int &&' cinit
// CHECK-NEXT:  `-CXXStaticCastExpr {{.*}} 'int' xvalue static_cast<int &&> <NoOp>
// CHECK-NEXT:    `-MemberExpr {{.*}} 'int' lvalue bitfield .a
// CHECK-NEXT:      `-DeclRefExpr {{.*}} 'A' lvalue ParmVar {{.*}} 'obj' 'A'
// CHECK:       VarDecl {{.*}} cr 'const int &&' cinit
// CHECK-NEXT:  `-CXXStaticCastExpr {{.*}} 'const int' xvalue static_cast<const int &&> <NoOp>
// CHECK-NEXT:    `-MemberExpr {{.*}} 'int' lvalue bitfield .a
// CHECK-NEXT:      `-DeclRefExpr {{.*}} 'A' lvalue ParmVar {{.*}} 'obj' 'A'
