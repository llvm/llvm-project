
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int glen;

struct T {
    char *buf __counted_by(glen); // expected-error{{count expression on struct field may only reference other fields of the same struct}}
    int len;
};

struct T_flex_1 {
    int dummy;
    int fam[__counted_by(glen)]; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
};


int glen2;
struct T_flex_2 {
    int dummy;
    int fam[__counted_by(glen2)]; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
};

void Test () {
    int len;
    struct S {
        int *__counted_by(len) buf; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
        int *__counted_by(glen) buf2; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
    };

    struct S_flex_1 {
        int dummy;
        int fam[__counted_by(len)]; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
    };

    struct S_flex_2 {
        int dummy;
        int fam[__counted_by(glen)]; // expected-error{{count expression on struct field may only reference other fields of the same struct}}
    };
}

void foo(int *__counted_by(glen) buf); // expected-error{{count expression in function declaration may only reference parameters of that function}}
void bar(int p[__counted_by(glen)]); // expected-error{{count expression in function declaration may only reference parameters of that function}}
