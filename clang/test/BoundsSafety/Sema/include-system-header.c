
// RUN: %clang_cc1 -fsyntax-only -DSYSTEM_HEADER -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -DSYSTEM_HEADER -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include "mock-header.h"

int main() {
    // In this configuration, `returns_pointer` returns an __unsafe_indexable
    // pointer, so it _does not_ convert to a __bidi_indexable pointer.
    int *p = returns_pointer(); // expected-error{{initializing 'int *__bidi_indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    return *p;
}
