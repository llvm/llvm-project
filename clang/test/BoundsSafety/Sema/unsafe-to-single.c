
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-unused-value -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-unused-value -verify %s

#include <ptrcheck.h>

struct SizedByData {
    void *__sized_by(len) data;
    unsigned len;
};

void unsafe_to_sizedby(int *__unsafe_indexable ptrUnsafe) {
    struct SizedByData st;
    st.data = ptrUnsafe; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}

void unsafe_to_single(int *__unsafe_indexable ptrUnsafe) {
    int *__single sp = ptrUnsafe; // expected-error-re{{initializing {{.+}} with an expression of incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}

struct EndedByData {
    char *end;
    char *__ended_by(end) begin;
};

void unsafe_to_endedby(struct EndedByData *st) {
    int *__unsafe_indexable ptrUnsafe;
    st->begin = ptrUnsafe; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    st->end = ptrUnsafe; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}

// expected-note@+1{{passing argument to parameter 'arg' here}}
void func_with_single(int *arg);
void unsafe_to_single_parameter(void) {
    int *__unsafe_indexable ptrUnsafe;
    func_with_single(ptrUnsafe); // expected-error-re{{passing '{{.+}}*__unsafe_indexable' to parameter of incompatible type {{.+}} casts away '__unsafe_indexable' qualifier}}
}

typedef int * myintptr_t;
void null_to_single(void) {
    myintptr_t __single dst = (myintptr_t)0;
}

void pointer_casts(int *__unsafe_indexable ptrUnsafe) {
    int *__single ptrSingle = (char *)ptrUnsafe; // expected-error-re{{initializing {{.+}} with an expression of incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    ptrSingle = (char *__unsafe_indexable)ptrSingle; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    ptrSingle = (char *__unsafe_indexable)ptrUnsafe; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}

void unsafe_to_single_nested(void) {
    int *__unsafe_indexable* ptrUnsafeNested;
    int **ptr2d = ptrUnsafeNested; // expected-error{{initializing 'int *__single*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__unsafe_indexable*__bidi_indexable'}}
}

void pointer_cast_2d(void) {
    int **ptr2d;
    char **cptr2d = (char**)ptr2d;
}

void pointer_cast_2d_unsafe(void) {
    int *__unsafe_indexable*ptr2d;
    // expected-error@+1{{initializing 'char *__single*__bidi_indexable' with an expression of incompatible nested pointer type 'char *__unsafe_indexable*__bidi_indexable'}}
    char **cptr2d = (char**)ptr2d;
}

void pointer_cast_2d_unsafe_explicit(void) {
    int **ptr2d;
    // expected-error@+1{{initializing 'char *__single*__bidi_indexable' with an expression of incompatible nested pointer type 'char *__unsafe_indexable*__bidi_indexable'}}
    char **cptr2d = (char *__unsafe_indexable*)ptr2d;
}

void pointer_cast_2d_unsafe_to_safe_explicit(void) {
    int *__unsafe_indexable*ptr2d;
    // expected-error@+1{{initializing 'char *__unsafe_indexable*__bidi_indexable' with an expression of incompatible nested pointer type 'char *__single*__bidi_indexable'}}
    char *__unsafe_indexable*cptr2d = (char *__single*)ptr2d;
}
