

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Foo {
    int *__bidi_indexable *__single ptrBoundPtrThin;
    // CHECK: FieldDecl {{.+}} ptrBoundPtrThin 'int *__bidi_indexable*__single'
};

typedef struct Foo Foo;

Foo *__bidi_indexable *__single Test (Foo *__single *__bidi_indexable argFooPtrThinPtrBound) {
    Foo *__single *__bidi_indexable localFooPtrThinPtrBound = argFooPtrThinPtrBound;
    Foo *__bidi_indexable *__single Res;
    return Res;
}
// CHECK: FunctionDecl {{.+}} Test 'Foo *__bidi_indexable*__single(Foo *__single*__bidi_indexable)'
// CHECK: VarDecl {{.+}} localFooPtrThinPtrBound 'Foo *__single*__bidi_indexable' cinit
// CHECK: VarDecl {{.+}} Res 'Foo *__bidi_indexable*__single'
