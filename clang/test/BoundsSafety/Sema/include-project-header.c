
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// expected-no-diagnostics

#include <ptrcheck.h>
#include "mock-header.h"

int main() {
    // In this configuration, `returns_pointer` returns a __single pointer, so
    // it converts to __bidi_indexable and dereferences properly.
    int *p = returns_pointer();
    return *p;
}
