

// RUN: %clang_cc1 -ast-dump -verify -fbounds-safety -isystem %S/SystemHeaders/include %s | FileCheck %s
// RUN: %clang_cc1 -E -verify -fbounds-safety -isystem %S/SystemHeaders/include -o %t.pp.c %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety %t.pp.c | FileCheck %s

// expected-no-diagnostics

#include <ptrcheck.h>
#include <default-attributes-in-preprocessed.h>

int *__single return_to_single_p(int *p) {
  return p;
}

// CHECK-LABEL: increment_unsafe_p 'void (int *)' inline
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used p 'int *'
// CHECK-NEXT: | `-CompoundStmt
// CHECK-NEXT: |   `-UnaryOperator {{.*}} 'int *' postfix '++'
// CHECK-NEXT: |     `-DeclRefExpr {{.*}} 'int *' lvalue ParmVar {{.*}} 'p' 'int *'

// CHECK-LABEL: return_to_single_p 'int *__single(int *__single)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used p 'int *__single'
// CHECK-NEXT:   `-CompoundStmt
// CHECK-NEXT:     `-ReturnStmt
// CHECK-NEXT:       `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
// CHECK-NEXT:         `-DeclRefExpr {{.*}} 'int *__single' lvalue ParmVar {{.*}} 'p' 'int *__single'
