
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

int foo(int *__counted_by(*len)* out_buf, int *len, int *dummy_len);

int bar() {
    int len;
    int dummy_len;
    int *__counted_by(len) ptr;
    foo(&*&*&ptr, &*&len, &dummy_len);
    foo(&*(&*&ptr), (&(*(int*)&len)), &dummy_len);
    // switched len and dummy_len
    foo(&*&ptr, &dummy_len, &*&len); // expected-error{{incompatible dynamic count pointer argument to parameter of type}}
    return 0;
}
