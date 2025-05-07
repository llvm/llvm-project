
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s


#include <ptrcheck.h>

int side_effect(int *buf);

int foo(int *__counted_by(*len) *buf, int *len) {
    int arr[10];
    // expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
    int *__counted_by(*len) *p;
    int *__counted_by(*len) q; // expected-error{{dereference operator in '__counted_by' is only allowed for function parameters}}
    int *other_len;
    int **normal_p = buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
    len = other_len; // expected-error{{not allowed to change out parameter used as dependent count expression of other parameter}}
    buf = normal_p; // expected-error{{not allowed to change out parameter with dependent count}}
    *buf = arr;
    *len = side_effect(*buf); // expected-error{{assignments to dependent variables should not have side effects between them}}
    side_effect(*buf);
    *len = side_effect(*buf);
    *buf = arr;
    return 0;
}

int test_inbuf_outlen(int *__counted_by(*len) buf, int *len) {
    int arr[10];
    // expected-error@+2{{parameter 'buf' with '__counted_by' attribute depending on an indirect count is implicitly read-only}}
    // expected-error@+1{{assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') 'buf' requires corresponding assignment to '*len'; add self assignment '*len = *len' if the value has not changed}}
    buf = arr;
    side_effect(0);
    *len = 10;
    return 0;
}

int test_outbuf_inlen(int *__counted_by(len) *buf, int len) {
    int arr[10];
    // expected-error@+1{{assignment to 'len' requires corresponding assignment to 'int *__single __counted_by(len)' (aka 'int *__single') '*buf'; add self assignment '*buf = *buf' if the value has not changed}}
    len = 10;
    side_effect(0);
    // expected-error@+1{{assignment to 'int *__single __counted_by(len)' (aka 'int *__single') '*buf' requires corresponding assignment to 'len'; add self assignment 'len = len' if the value has not changed}}
    *buf = arr;
    return 0;
}

int foo_nullable(int *__counted_by_or_null(*len) *buf, int *len) {
    int arr[10];
    // expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
    int *__counted_by_or_null(*len) *p;
    int *__counted_by_or_null(*len) q; // expected-error{{dereference operator in '__counted_by_or_null' is only allowed for function parameters}}
    int *other_len;
    int **normal_p = buf; // expected-error{{pointer with '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
    len = other_len; // expected-error{{not allowed to change out parameter used as dependent count expression of other parameter}}
    buf = normal_p; // expected-error{{not allowed to change out parameter with dependent count}}
    *buf = arr;
    *len = side_effect(*buf); // expected-error{{assignments to dependent variables should not have side effects between them}}
    side_effect(*buf);
    *len = side_effect(*buf);
    *buf = arr;
    return 0;
}

int test_inbuf_outlen_nullable(int *__counted_by_or_null(*len) buf, int *len) {
    int arr[10];
    // expected-error@+2{{parameter 'buf' with '__counted_by_or_null' attribute depending on an indirect count is implicitly read-only}}
    // expected-error@+1{{assignment to 'int *__single __counted_by_or_null(*len)' (aka 'int *__single') 'buf' requires corresponding assignment to '*len'; add self assignment '*len = *len' if the value has not changed}}
    buf = arr;
    side_effect(0);
    *len = 10;
    return 0;
}

int test_outbuf_inlen_nullable(int *__counted_by_or_null(len) *buf, int len) {
    int arr[10];
    // expected-error@+1{{assignment to 'len' requires corresponding assignment to 'int *__single __counted_by_or_null(len)' (aka 'int *__single') '*buf'; add self assignment '*buf = *buf' if the value has not changed}}
    len = 10;
    side_effect(0);
    // expected-error@+1{{assignment to 'int *__single __counted_by_or_null(len)' (aka 'int *__single') '*buf' requires corresponding assignment to 'len'; add self assignment 'len = len' if the value has not changed}}
    *buf = arr;
    return 0;
}
