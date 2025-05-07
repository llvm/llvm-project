

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>
#include "compound-literal-pointer.h"

// CHECK: |-FunctionDecl {{.+}} return_ptr_nonptr
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *__bidi_indexable' lvalue
int *return_ptr_nonptr(void) {
    return (int *){ 0 };
}

// CHECK: |-FunctionDecl {{.+}} return_ptr_single
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *__single' lvalue
int *return_ptr_single(int *__single p) {
    return (int *){ p };
}

// CHECK: |-FunctionDecl {{.+}} return_ptr_indexable
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *__indexable' lvalue
int *return_ptr_indexable(int *__indexable p) {
    return (int *){ p };
}

// CHECK: |-FunctionDecl {{.+}} return_ptr_bidi_indexable
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *__bidi_indexable' lvalue
int *return_ptr_bidi_indexable(int *__bidi_indexable p) {
    return (int *){ p };
}

// CHECK: |-FunctionDecl {{.+}} return_ptr_counted_by
// CHECK: | `-CompoundLiteralExpr {{.+}} 'int *__bidi_indexable' lvalue
int *return_ptr_counted_by(int *__counted_by(n) p, int n) {
    return (int *){ p };
}

// CHECK: `-FunctionDecl {{.+}} return_ptr_unsafe_indexable
// CHECK:  `-CompoundLiteralExpr {{.+}} 'int *__unsafe_indexable' lvalue
int *__unsafe_indexable return_ptr_unsafe_indexable(int *__unsafe_indexable p) {
    return (int *){ p };
}
