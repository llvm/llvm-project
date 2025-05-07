
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
int foo(int *__counted_by(*out_n) *out_buf, int *out_n, int *no_out_n);


int main() {
    int n;
    int *__counted_by(n) buf;
    int *p = 0;
    int *__single plain_buf;
    foo(&buf, &n, p);
    foo(&buf, p, &n); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
    foo(&plain_buf, &n, p); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
    return 0;
}
