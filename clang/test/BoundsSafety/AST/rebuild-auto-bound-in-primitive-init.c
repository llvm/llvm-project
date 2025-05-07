

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

int main() {
    int *local;
    int *copy = local;
    int primitive = *local;
    return primitive;
}

// CHECK: `-FunctionDecl {{.+}} main 'int ()'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} local 'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} copy 'int *__bidi_indexable'{{.*}} cinit
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable'{{.*}} <LValueToRValue>
// CHECK:     |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable'{{.*}} lvalue Var {{.+}} 'local' 'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} used primitive 'int' cinit
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     `-UnaryOperator {{.+}} 'int' lvalue prefix '*' cannot overflow
// CHECK:     |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable'{{.*}} <LValueToRValue>
// CHECK:     |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable'{{.*}} lvalue Var {{.+}} 'local' 'int *__bidi_indexable'
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.+}} 'int' lvalue Var {{.+}} 'primitive' 'int'
