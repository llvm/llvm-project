

// FIXME: rdar://69452444
// RUN: not %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s --check-prefix=CHECK-M2
// RUN: not %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s --check-prefix=CHECK-M2
#include <ptrcheck.h>


void Test(void) {
    int *localAddrTaken;
    int **pp = &localAddrTaken;
    int *localNoAddrTaken;
    int *__bidi_indexable localBoundAddrTaken;
    int *__bidi_indexable *boundPP = &localBoundAddrTaken;
    void (*fptr)(void) = &Test;
    void (*fptr2)(void) = Test;
}

// FIXME: rdar://69452444
// CHECK:      `-FunctionDecl {{.+}} Test 'void (void)'
// CHECK-NEXT:  `-CompoundStmt
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | `-VarDecl {{.+}} used localAddrTaken 'int *__single'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | `-VarDecl {{.+}} pp 'int *__single*__bidi_indexable' cinit
// CHECK-NEXT:    |   `-UnaryOperator {{.+}} 'int *__single*__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:    |     `-DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'localAddrTaken' 'int *__single'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | `-VarDecl {{.+}} localNoAddrTaken 'int *__bidi_indexable'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | `-VarDecl {{.+}} localBoundAddrTaken 'int *__bidi_indexable'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | `-VarDecl {{.+}} boundPP 'int *__bidi_indexable*__bidi_indexable' cinit
// CHECK-NEXT:    |   `-UnaryOperator {{.+}} 'int *__bidi_indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:    |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{0x[a-z0-9]*}} 'localBoundAddrTaken' 'int *__bidi_indexable'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | `-VarDecl {{.+}} fptr 'void (*__single)(void)':'void (*__single)(void)' cinit
// CHECK-NEXT:    |   `-UnaryOperator {{.+}} 'void (*__single)(void)' prefix '&' cannot overflow
// CHECK-NEXT:    |     `-DeclRefExpr {{.+}} 'void (void)' Function {{.+}} 'Test' 'void (void)'
// CHECK-NEXT:    `-DeclStmt
// CHECK-NEXT:      `-VarDecl {{.+}} fptr2 'void (*__single)(void)':'void (*__single)(void)' cinit
// CHECK-NEXT:        `-ImplicitCastExpr {{.+}} 'void (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT:          `-DeclRefExpr {{.+}} 'void (void)' Function {{.+}} 'Test' 'void (void)'

// CHECK-M2: `-FunctionDecl
// CHECK-M2:   `-CompoundStmt
// CHECK-M2:     |-DeclStmt
// CHECK-M2:     | `-VarDecl {{.+}} used localAddrTaken 'int *__bidi_indexable'
// CHECK-M2:     |-DeclStmt
// CHECK-M2:     | `-VarDecl {{.+}} pp 'int *__single*__bidi_indexable'
// CHECK-M2:     |-DeclStmt
// CHECK-M2:     | `-VarDecl {{.+}} localNoAddrTaken 'int *__bidi_indexable'
// CHECK-M2:     |-DeclStmt
// CHECK-M2:     | `-VarDecl {{.+}} used localBoundAddrTaken 'int *__bidi_indexable'
// CHECK-M2:     |-DeclStmt
// CHECK-M2:     | `-VarDecl {{.+}} boundPP 'int *__bidi_indexable*__bidi_indexable'
// CHECK-M2:     |   `-UnaryOperator {{.+}} 'int *__bidi_indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK-M2:     |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue Var {{.+}} 'localBoundAddrTaken' 'int *__bidi_indexable'
// CHECK-M2:     |-DeclStmt
// CHECK-M2:     | `-VarDecl {{.+}} fptr 'void (*__single)(void)' cinit
// CHECK-M2:     |   `-UnaryOperator {{.+}} 'void (*__single)(void)' prefix '&' cannot overflow
// CHECK-M2:     |     `-DeclRefExpr {{.+}} 'void (void)' Function {{.+}} 'Test' 'void (void)'
// CHECK-M2:     `-DeclStmt
// CHECK-M2:       `-VarDecl {{.+}} fptr2 'void (*__single)(void)' cinit
// CHECK-M2:         `-ImplicitCastExpr {{.+}} 'void (*__single)(void)' <FunctionToPointerDecay>
// CHECK-M2:           `-DeclRefExpr {{.+}} 'void (void)' Function {{.+}} 'Test' 'void (void)'
