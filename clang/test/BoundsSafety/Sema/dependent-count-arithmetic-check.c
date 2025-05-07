
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
int foo(int *__counted_by(*out_n + 1) *out_buf, int *out_n); // expected-error{{invalid argument expression to bounds attribute}}

int bar(int *__counted_by(*out_n) *out_buf, int *out_n);


struct T {
    int *__counted_by(l) buf;
    int l;
};


int main() {
    int n = -1;
    int *__counted_by(n + 1) buf;
    // XXX: this error message is misleading because `foo` had an attribute error and
    // thus the function parameter does not have the proper attribute to check.
    // expected-error@+1{{passing address of 'n' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
    foo(&buf, &n);

    bar(&buf, &n); // expected-error{{incompatible count expression '*out_n' vs. 'n + 1' in argument to function}}
    return 0;
}
