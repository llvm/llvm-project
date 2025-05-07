

// RUN: %clang_cc1 -fbounds-safety -fsyntax-only -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -verify %s

#include <ptrcheck.h>

int foo(int * __indexable ptr, unsigned idx) {
    ptr -= idx;
    // expected-error@-1 {{decremented indexable pointer 'ptr' is out of bounds}}
    return *ptr;
}

int main() {
    int a;
    int * __indexable p = &a;

    return foo(p, -1);
}
