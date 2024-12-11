

// RUN: %clang_cc1 -fbounds-safety -triple arm64 -verify %s

#include <ptrcheck.h>
#include <stddef.h>

int *__bidi_indexable to_bidi(int * arg) {
    int len = -1;
    // expected-error@+1{{possibly initializing 'p' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') and count value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __counted_by_or_null(len) p = arg;
    return p;
}

int *__bidi_indexable to_bidi_literal_count(int * arg) {
    // expected-error@+1{{possibly initializing 'p' of type 'int *__single __counted_by_or_null(-1)' (aka 'int *__single') and count value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __counted_by_or_null(-1) p = arg;
    return p;
}

int *__bidi_indexable to_bidi_const_count(int * arg) {
    const int len = -1;
    // expected-error@+1{{possibly initializing 'p' of type 'int *__single __counted_by_or_null(-1)' (aka 'int *__single') and count value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __counted_by_or_null(len) p = arg;
    return p;
}

void back_and_forth_to_bidi(int * __bidi_indexable arg) {
    int len = -1;
    int * __counted_by_or_null(len) p = NULL;
    arg = p;
    p = arg;
    len = len;
    arg = p;
}

void foo(int *__counted_by(len) p, int len);

void to_counted_by_arg(int * arg) {
    int len = -1;
    int * __counted_by_or_null(len) p = NULL;
    foo(p, len);
    int len2 = -1;
    // expected-error@+1{{possibly initializing 'p2' of type 'int *__single __counted_by_or_null(len2)' (aka 'int *__single') and count value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __counted_by_or_null(len2) p2 = arg;
    foo(p2, len2);
    // expected-error@+1{{negative count value of -1 for 'p' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
    foo(NULL, -1);
    foo(NULL, 0);
    // expected-error@+1{{negative count value of -1 for 'p' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
    foo(arg, -1);
    foo(arg, 0);
}

void bar(int *__counted_by_or_null(len) p, int len);

void to_counted_by_or_null_arg(int * arg) {
    // expected-error@+1{{possibly passing non-null to parameter 'p' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') with count value of -1; explicitly pass null to remove this warning}}
    bar(arg, -1);
    bar(arg, 0);
    bar(NULL, -1);
    bar(NULL, 0);
}

int * __counted_by_or_null(len) nullable_ret(int len);
int * __counted_by(len) nonnullable_ret(int len);

void ret_values() {
    int len = -1;
    nullable_ret(-1);
    nullable_ret(len);
    nonnullable_ret(-1);
    nonnullable_ret(len);
    len = 0;
    nullable_ret(len);
    nonnullable_ret(len);
}

struct offset_fam {
    int len;
    // expected-note@+1 5{{initialized flexible array member 'buf' is here}}
    int buf[__counted_by(len-1)];
};

struct offset_nonnullable {
    int len;
    int * __counted_by(len-1) buf;
};

struct offset_nullable {
    int len;
    int * __counted_by_or_null(len-1) buf;
};

void struct_fields(int * arg) {
    struct offset_fam fam1; // rdar://127523062 We should not allow stack allocated flexible array members
    // expected-error@+1{{flexible array member is initialized with 0 elements, but count value is initialized to -1}}
    struct offset_fam fam2 = {};
    // expected-error@+1{{flexible array member is initialized with 0 elements, but count value is initialized to -1}}
    struct offset_fam fam3 = { .buf = {} };
    // expected-error@+1{{flexible array member is initialized with 1 element, but count value is initialized to -1}}
    static struct offset_fam fam4 = { .buf = {1} };
    static struct offset_fam fam5 = { .buf = {}, .len = 1};
    static struct offset_fam fam6 = { .len = 1, .buf = {} };
    // expected-error@+1{{flexible array member is initialized with 1 element, but count value is initialized to 0}}
    static struct offset_fam fam7 = { .len = 1, .buf = {1} };
    // expected-error@+1{{flexible array member is initialized with 0 elements, but count value is initialized to 1}}
    static struct offset_fam fam8 = { .len = 2, .buf = {} };
    static struct offset_fam fam9 = { .len = 2, .buf = {1} };

    // expected-error@+1{{negative count value of -1 for 'nn1.buf' of type 'int *__single __counted_by(len - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn1;
    // expected-error@+1{{negative count value of -1 for 'nn2.buf' of type 'int *__single __counted_by(len - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn2 = {};
    // expected-error@+1{{negative count value of -1 for 'nn3.buf' of type 'int *__single __counted_by(len - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn3 = { .buf = arg };
    struct offset_nonnullable nn4 = { .len = 1, .buf = arg };
    struct offset_nonnullable nn5 = { .len = 2, .buf = arg };

    struct offset_nullable n1;
    struct offset_nullable n2 = {};
    // expected-error@+1{{possibly initializing 'n3.buf' of type 'int *__single __counted_by_or_null(len - 1)' (aka 'int *__single') and count value of -1 with non-null; explicitly initialize null to remove this warning}}
    struct offset_nullable n3 = { .buf = arg };
    struct offset_nullable n4 = { .len = 1, .buf = arg };
    struct offset_nullable n5 = { .len = 2, .buf = arg };
}
