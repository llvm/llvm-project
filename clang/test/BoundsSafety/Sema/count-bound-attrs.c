
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void Test(void) {
    int len;
    int *__bidi_indexable __counted_by(len) boundCount; // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
    int len2;
    int *__single __counted_by(len2) thinCount;
    int len3;
    int *__indexable __counted_by(len3) arrayCount; // expected-error{{pointer cannot be '__counted_by' and '__indexable' at the same time}}
    int len5;
    int *__unsafe_indexable __counted_by(len5) unsafeCount;
    int len6;
    int *__counted_by(len6) justCount;
    int len7;
    int *__counted_by(len7) __counted_by(len7+1) thinCountCount; // expected-error{{pointer cannot have more than one count attribute}}

    int len8;
    int *__counted_by(len8) __single countThin;
    int len9;
    int *__counted_by(len9) __bidi_indexable countBound; // expected-error{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
    int len10;
    int *__counted_by(len10) __indexable countArray; // expected-error{{pointer cannot be '__counted_by' and '__indexable' at the same time}}
    int len12;
    int *__counted_by(len12) __unsafe_indexable countUnsafe;

    int len13;
    void *__counted_by(len13) countVoid; // expected-error{{cannot apply '__counted_by' attribute to 'void *' because 'void' has unknown size; did you mean to use '__sized_by' instead?}}
    int len14;
    void *__sized_by(len14) byteCountVoid;
}
