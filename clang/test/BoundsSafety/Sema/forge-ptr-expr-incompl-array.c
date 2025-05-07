

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct Foo {
    int f;
    int vla[];
};

void Test() {
    struct Foo f;
    (void) __unsafe_forge_bidi_indexable(int *, f.vla, 0);
    (void) __unsafe_forge_single(int *, f.vla);
    (void) __unsafe_forge_terminated_by(int *, f.vla, 255);
}

// expected-no-diagnostics
