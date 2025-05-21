// RUN: %clang_cc1 %s -verify -ast-dump | FileCheck %s

// expected-no-diagnostics

//      CHECK: ExtVectorType {{.*}} 'int __attribute__((ext_vector_type(4)))' 4
// CHECK-NEXT: BuiltinType {{.*}} 'int'
int x __attribute__((ext_vector_type(4)));
using ExtVecType = decltype(x);

// CHECK: FunctionDecl {{.*}} 'int () __attribute__((ext_vector_type(4)))'
int __attribute__((ext_vector_type(4))) foo() { return x; }
//      CHECK:  CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int __attribute__((ext_vector_type(4)))' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int __attribute__((ext_vector_type(4)))' lvalue Var {{.*}} 'x' 'int __attribute__((ext_vector_type(4)))'
