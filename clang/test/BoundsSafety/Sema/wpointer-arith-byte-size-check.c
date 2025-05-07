
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wpointer-arith -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wpointer-arith -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>

struct S {
    int *__sized_by(l) bp;
    int l;
};

int main() {
    struct S s;

    int arr[10];
    s.bp = arr;
    s.l = 10;

    return 0;
}
