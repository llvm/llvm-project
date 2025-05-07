// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -O0 -fbounds-safety -S %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -S %s -o - | FileCheck %s

#include <ptrcheck.h>

struct Foo {
    char *__bidi_indexable inner;
};
int globalCh;

void bar(void) {
    int localCh;

    struct Foo local = {
        .inner = &globalCh
    };
}

// CHECK: local:
// CHECK:   .quad	_globalCh
// CHECK:   .quad	_globalCh+4
// CHECK:   .quad	_globalCh


