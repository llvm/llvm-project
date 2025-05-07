
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

struct T1 {
    int len;
    int len2;
    int *__counted_by(len) __counted_by(len+1) buf; // expected-error{{pointer cannot have more than one count attribute}}
    int *__counted_by(len + len2) buf2;
    int *__bidi_indexable __counted_by(len) buf3; // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
    int *__counted_by(len) __bidi_indexable buf4; // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
};

// expected-note@+1{{'foo' declared here}}
int foo(void);

struct T2 {
    float flen;
    int *__counted_by(flen) buf1; // expected-error{{attribute requires an integer type argument}}
    int *__counted_by(foo()) buf2; // expected-error{{argument of '__counted_by' attribute can only reference function with 'const' attribute}}
};

int glen;

int Test(int *__counted_by(glen) ptr); // expected-error{{count expression in function declaration may only reference parameters of that function}}

int *__bidi_indexable __counted_by(glen) glob_buf1; // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
int *__counted_by(glen) __bidi_indexable glob_buf2; // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}

void foo1(int *__bidi_indexable __counted_by(glen) glob_buf1); // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void foo2(int *__counted_by(glen) __bidi_indexable glob_buf2); // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
