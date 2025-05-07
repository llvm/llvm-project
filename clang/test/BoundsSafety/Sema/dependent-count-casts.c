
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s


#include <ptrcheck.h>
// rdar://70688465
int foo(int *__counted_by(len) *buf, int len) {
    int **local_buf = buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}

    return 0;
}

int bar(int *__counted_by_or_null(len) *buf, int len) {
    int **local_buf = buf; // expected-error{{pointer with '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}

    return 0;
}

int main() {
    int *__single buf;
    foo(&buf, 10); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
    bar(&buf, 10); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
    return 0;
}
