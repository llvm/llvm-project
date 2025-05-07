
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-unused-value -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-unused-value -verify %s

#include <ptrcheck.h>

typedef struct {
    void *vptr;
    int valid;
} S;

void foo() {
    int *__unsafe_indexable ptrUnsafe;
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    int *ptrAuto = (int *__bidi_indexable)(ptrUnsafe);
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    int *__indexable ptrIndex = (char *__bidi_indexable)(ptrUnsafe);
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    int *__bidi_indexable ptrBidiIndex = (void *__indexable)(ptrUnsafe);
    int *ptrAuto2 = (void *__indexable)0;
    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety}}
    int *ptrAuto3 = (void *__indexable)0xdead;

    (void *__bidi_indexable) ptrIndex;
    (void *__indexable) ptrIndex;
    (void *__bidi_indexable) ptrBidiIndex;
    (void *__indexable) ptrBidiIndex;
    (int *__bidi_indexable) ptrIndex;
    (int *__indexable) ptrIndex;
    (int *__bidi_indexable) ptrBidiIndex;
    (int *__indexable) ptrBidiIndex;
    (S *__bidi_indexable) ptrIndex;
    (S *__indexable) ptrIndex;
    (S *__bidi_indexable) ptrBidiIndex;
    (S *__indexable) ptrBidiIndex;

    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety}}
    ptrAuto = (char *__bidi_indexable) (0xdead);
    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety}}
    (char *__single) (0xdead);
    (char *) (0xdead);
    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety}}
    (char *__bidi_indexable) 0xdead;
    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety}}
    (S *__indexable) 0xdead;

    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    (void *__bidi_indexable) ptrUnsafe;
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    (void *__indexable) ptrUnsafe;
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    (int *__bidi_indexable) ptrUnsafe;
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    (int *__indexable) ptrUnsafe;
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    (S *__bidi_indexable) ptrUnsafe;
    // expected-error-re@+1{{casting 'int *__unsafe_indexable' to incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
    (S *__indexable) ptrUnsafe;
}
