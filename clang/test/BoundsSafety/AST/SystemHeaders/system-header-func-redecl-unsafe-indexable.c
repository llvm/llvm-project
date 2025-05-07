
// RUN: %clang_cc1 -fsyntax-only -ast-dump -fbounds-safety -I %S/include %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump -fbounds-safety -I %S/include -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>
#include <system-header-func-decl.h>

int *__unsafe_indexable foo(int *__unsafe_indexable *__unsafe_indexable);

void bar(void) {
    int *__unsafe_indexable s;
    foo(&s);
}
// CHECK: `-FunctionDecl {{.+}} bar 'void (void)'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} used s 'int *__unsafe_indexable'
// CHECK:     `-CallExpr {{.+}} 'int *__unsafe_indexable'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable(*__single)(int *__unsafe_indexable*__unsafe_indexable)' <FunctionToPointerDecay>
// CHECK:       | `-DeclRefExpr {{.+}} 'int *__unsafe_indexable(int *__unsafe_indexable*__unsafe_indexable)' Function {{.+}} 'foo' 'int *__unsafe_indexable(int *__unsafe_indexable*__unsafe_indexable)'
// CHECK:       `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable*__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK:         `-UnaryOperator {{.+}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK:           `-DeclRefExpr {{.+}} 'int *__unsafe_indexable' lvalue Var {{.+}} 's' 'int *__unsafe_indexable'
