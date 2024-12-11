

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fbounds-attributes-objc-experimental -verify %s

#include <ptrcheck.h>

struct Foo {
    int f;
    int vla[];
};

void Test() {
    struct Foo f;
    (void) f.vla;
    // expected-warning@-1{{accessing elements of an unannotated incomplete array always fails at runtime}}
}
