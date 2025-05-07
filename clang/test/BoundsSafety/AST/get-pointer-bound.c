

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

void bidi_indexable(int *__bidi_indexable p) {
    int *q = __builtin_get_pointer_lower_bound(p);
    int *r = __builtin_get_pointer_upper_bound(p);
}

// CHECK: |   |-DeclStmt {{.+}}
// CHECK: |   | `-VarDecl {{.+}} q 'int *__bidi_indexable' cinit
// CHECK: |   |   `-GetBoundExpr {{.+}} 'int *__bidi_indexable' lower
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: |   |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'p' 'int *__bidi_indexable'
// CHECK: |   `-DeclStmt {{.+}}
// CHECK: |     `-VarDecl {{.+}} r 'int *__bidi_indexable' cinit
// CHECK: |       `-GetBoundExpr {{.+}} 'int *__bidi_indexable' upper
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'p' 'int *__bidi_indexable'

void fwd_indexable(int *__indexable p) {
    int *q = __builtin_get_pointer_lower_bound(p);
    int *r = __builtin_get_pointer_upper_bound(p);
}

// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} q 'int *__bidi_indexable' cinit
// CHECK: |   |   `-GetBoundExpr {{.+}} 'int *__bidi_indexable' lower
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK: |   |         `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl {{.+}} r 'int *__bidi_indexable' cinit
// CHECK: |       `-GetBoundExpr {{.+}} 'int *__bidi_indexable' upper
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |           `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK: |             `-DeclRefExpr {{.+}} 'int *__indexable' lvalue ParmVar {{.+}} 'p' 'int *__indexable'

void single(int *__single p) {
    int *q = __builtin_get_pointer_lower_bound(p);
    int *r = __builtin_get_pointer_upper_bound(p);
}

// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} q 'int *__bidi_indexable' cinit
// CHECK: |   |   `-GetBoundExpr {{.+}} 'int *__bidi_indexable' lower
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: |   |         `-DeclRefExpr {{.+}} 'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single'
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl {{.+}} r 'int *__bidi_indexable' cinit
// CHECK: |       `-GetBoundExpr {{.+}} 'int *__bidi_indexable' upper
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |           `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: |             `-DeclRefExpr {{.+}} 'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single'

void array(void) {
    int array[10];
    int *q = __builtin_get_pointer_lower_bound(array);
    int *r = __builtin_get_pointer_upper_bound(array);
}

// CHECK: |-FunctionDecl {{.+}} array 'void (void)'
// CHECK: | `-CompoundStmt {{.+}}
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} used array 'int[10]'
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.+}} q 'int *__bidi_indexable' cinit
// CHECK: |   |   `-GetBoundExpr {{.+}} 'int *__bidi_indexable' lower
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |   |       `-DeclRefExpr {{.+}} 'int[10]' lvalue Var {{.+}} 'array' 'int[10]'
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl {{.+}} col:10 r 'int *__bidi_indexable' cinit
// CHECK: |       `-GetBoundExpr {{.+}} 'int *__bidi_indexable' upper
// CHECK: |         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |           `-DeclRefExpr {{.+}} 'int[10]' lvalue Var {{.+}} 'array' 'int[10]'

int endOfFile;
