

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

void foo(int *a) {
// CHECK: -VarDecl {{.+}} b 'int *__single*__bidi_indexable' cinit
// CHECK: -VarDecl {{.+}} c 'int *__single*__bidi_indexable' cinit
    int *__single *b = &a;
    __auto_type c = &a;

    // PointerTypeLoc should have the SourceLocation of __single, __indexable,
    // __bidi_indexable, __unsafe_indexable:
    // without it, TreeTransform::TransformPointerType doesn't know what
    // attributes to use. Keeping this as CHECK-NOT as a reminder to update
    // this test when it's fixed.

// CHECK-NOT: -VarDecl {{.+}} d 'int *__single*__bidi_indexable' cinit
// CHECK-NOT: -VarDecl {{.+}} e 'int *__single*__bidi_indexable' cinit
    __auto_type *d = &a;
    __auto_type *__single *e = &a;
}
