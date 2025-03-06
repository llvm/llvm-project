// RUN: %clang_cc1 %s -verify -ast-dump | FileCheck %s

// expected-no-diagnostics

// CHECK: VarDecl {{.*}} x 'int __attribute__((ext_vector_type(4)))'
int x __attribute__((ext_vector_type(4)));

// CHECK: FunctionDecl {{.*}} 'int () __attribute__((ext_vector_type(4)))'
int __attribute__((ext_vector_type(4))) foo() { return x; }
// CHECK:  CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int __attribute__((ext_vector_type(4)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int __attribute__((ext_vector_type(4)))' lvalue Var {{.*}} 'x' 'int __attribute__((ext_vector_type(4)))'
