

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void Foo(int *__counted_by(len) buf, char *dummy, int len) {}
void Bar(int len, char *dummy, int *__counted_by(len) buf) {}
void Baz(int *__counted_by(*out_len)* out_buf, char *dummy, int *out_len) {}
void Qux(int *out_len, char *dummy, int *__counted_by(*out_len)* out_buf) {}

unsigned long get_len(void *__bidi_indexable ptr);
unsigned long trap_if_bigger_than_max(unsigned long len);

char *side_effect_cp();
int *__bidi_indexable side_effect_ip();

int Test() {
    int arr[10];
    char *cp;
    int len;

    Foo(arr, cp, 10);
    Foo(side_effect_ip(), side_effect_cp(), 10); // ok

    Foo(side_effect_ip(), cp, len); // ok

    Foo(arr, side_effect_cp(), len); // ok

    Bar(len, side_effect_cp(), side_effect_ip()); // ok
    Bar(get_len(arr), cp, arr); // ok

    int cnt;
    int *__counted_by(cnt) ptr;
    // Out parameters are okay because it only takes already type safe arguments.
    Baz(&ptr, side_effect_cp(), &cnt); // ok
    Baz(&ptr, cp, &cnt); // ok
    Qux(&cnt, side_effect_cp(), &ptr); // ok
    return 0;
}

// expected-no-diagnostics
