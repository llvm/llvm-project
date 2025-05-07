

// RUN: %clang_cc1 -ast-dump -fbounds-safety -verify %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s | FileCheck %s
#include <ptrcheck.h>

__ptrcheck_abi_assume_single()
int *FSingle(int *x) {
  int **y = &x;
  return *y;
}
// CHECK: |-FunctionDecl {{.+}} FSingle 'int *__single(int *__single)'
// CHECK: | |-ParmVarDecl {{.+}} used x 'int *__single'
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} used y 'int *__single*__bidi_indexable' cinit
// CHECK: |   |   `-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
// CHECK: |   |     `-DeclRefExpr {{.+}} 'int *__single' lvalue ParmVar {{.+}} 'x' 'int *__single'
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: |       `-UnaryOperator {{.+}} 'int *__single' lvalue prefix '*' cannot overflow
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__single*__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} 'int *__single*__bidi_indexable' lvalue Var {{.+}} 'y' 'int *__single*__bidi_indexable'



__ptrcheck_abi_assume_indexable()
int *FIndexable(int *x) {
  int **y = &x;
  return *y;
}
// CHECK: |-FunctionDecl {{.+}} FIndexable 'int *__indexable(int *__indexable)'
// CHECK: | |-ParmVarDecl {{.+}} used x 'int *__indexable'
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} used y 'int *__indexable*__bidi_indexable' cinit
// CHECK: |   |   `-UnaryOperator {{.+}} 'int *__indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK: |   |     `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'x' 'int *__indexable'
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK: |       `-UnaryOperator {{.+}} 'int *__indexable' lvalue prefix '*' cannot overflow
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__indexable*__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} 'int *__indexable*__bidi_indexable' lvalue Var {{.+}} 'y' 'int *__indexable*__bidi_indexable'


__ptrcheck_abi_assume_bidi_indexable()
int *FBidiIndexable(int *x) {
  int **y = &x;
  return *y;
}
// CHECK: |-FunctionDecl {{.+}} FBidiIndexable 'int *__bidi_indexable(int *__bidi_indexable)'
// CHECK: | |-ParmVarDecl {{.+}} used x 'int *__bidi_indexable'
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} used y 'int *__bidi_indexable*__bidi_indexable' cinit
// CHECK: |   |   `-UnaryOperator {{.+}} 'int *__bidi_indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK: |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'x' 'int *__bidi_indexable'
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: |       `-UnaryOperator {{.+}} 'int *__bidi_indexable' lvalue prefix '*' cannot overflow
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable*__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} 'int *__bidi_indexable*__bidi_indexable' lvalue Var {{.+}} 'y' 'int *__bidi_indexable*__bidi_indexable'



__ptrcheck_abi_assume_unsafe_indexable()
int *FUnsafeIndexable(int *x) {
  int **y = &x;
  return *y;
}
// CHECK: `-FunctionDecl {{.+}} FUnsafeIndexable 'int *__unsafe_indexable(int *__unsafe_indexable)'
// CHECK:   |-ParmVarDecl {{.+}} used x 'int *__unsafe_indexable'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} used y 'int *__unsafe_indexable*__unsafe_indexable' cinit
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable*__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     `-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK:     |       `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue ParmVar {{.+}} 'x' 'int *__unsafe_indexable'
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// CHECK:         `-UnaryOperator {{.+}} 'int *__unsafe_indexable' lvalue prefix '*' cannot overflow
// CHECK:           `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable*__unsafe_indexable' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} 'int *__unsafe_indexable*__unsafe_indexable' lvalue Var {{.+}} 'y' 'int *__unsafe_indexable*__unsafe_indexable'

// expected-no-diagnostics
