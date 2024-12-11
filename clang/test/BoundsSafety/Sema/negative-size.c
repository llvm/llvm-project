

// RUN: %clang_cc1 -fbounds-safety -triple arm64 -verify %s

#include <ptrcheck.h>
#include <stddef.h>

int *__bidi_indexable to_bidi(int * arg) {
    int size = -1;
    // expected-error@+1{{possibly initializing 'p' of type 'int *__single __sized_by_or_null(size)' (aka 'int *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __sized_by_or_null(size) p = arg;
    return p;
}

int *__bidi_indexable to_bidi_literal_size(int * arg) {
    // expected-error@+1{{possibly initializing 'p' of type 'int *__single __sized_by_or_null(-1)' (aka 'int *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __sized_by_or_null(-1) p = arg;
    return p;
}

int *__bidi_indexable to_bidi_const_size(int * arg) {
    const int size = -1;
    // expected-error@+1{{possibly initializing 'p' of type 'int *__single __sized_by_or_null(-1)' (aka 'int *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __sized_by_or_null(size) p = arg;
    return p;
}

void back_and_forth_to_bidi(int * __bidi_indexable arg) {
    int size = -1;
    int * __sized_by_or_null(size) p = NULL;
    arg = p;
    p = arg;
    size = size;
    arg = p;
}

void foo(int *__sized_by(size) p, int size);

void to_sized_by_arg(int * arg) {
    int size = -1;
    int * __sized_by_or_null(size) p = NULL;
    foo(p, size);
    int size2 = -1;
    // expected-error@+1{{possibly initializing 'p2' of type 'int *__single __sized_by_or_null(size2)' (aka 'int *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
    int * __sized_by_or_null(size2) p2 = arg;
    foo(p2, size2);
    // expected-error@+1{{negative size value of -1 for 'p' of type 'int *__single __sized_by(size)' (aka 'int *__single')}}
    foo(NULL, -1);
    foo(NULL, 0);
    // expected-error@+1{{negative size value of -1 for 'p' of type 'int *__single __sized_by(size)' (aka 'int *__single')}}
    foo(arg, -1);
    foo(arg, 0);
}

void bar(int *__sized_by_or_null(size) p, int size);

void to_sized_by_or_null_arg(int * arg) {
    // expected-error@+1{{possibly passing non-null to parameter 'p' of type 'int *__single __sized_by_or_null(size)' (aka 'int *__single') with size value of -1; explicitly pass null to remove this warning}}
    bar(arg, -1);
    bar(arg, 0);
    bar(NULL, -1);
    bar(NULL, 0);
}

int * __sized_by_or_null(size) nullable_ret(int size);
int * __sized_by(size) nonnullable_ret(int size);

void ret_values() {
    int size = -1;
    nullable_ret(-1);
    nullable_ret(size);
    nonnullable_ret(-1);
    nonnullable_ret(size);
    size = 0;
    nullable_ret(size);
    nonnullable_ret(size);
}

struct offset_nonnullable {
    int size;
    int * __sized_by(size-1) buf;
};

struct offset_nullable {
    int size;
    int * __sized_by_or_null(size-1) buf;
};

void struct_fields(int * arg) {
    // expected-error@+1{{negative size value of -1 for 'nn1.buf' of type 'int *__single __sized_by(size - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn1;
    // expected-error@+1{{negative size value of -1 for 'nn2.buf' of type 'int *__single __sized_by(size - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn2 = {};
    // expected-error@+1{{negative size value of -1 for 'nn3.buf' of type 'int *__single __sized_by(size - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn3 = { .buf = arg };
    struct offset_nonnullable nn4 = { .size = 1, .buf = arg };
    struct offset_nonnullable nn5 = { .size = 5, .buf = arg };
    // expected-error@+1{{negative size value of -1 for 'nn6.buf' of type 'int *__single __sized_by(size - 1)' (aka 'int *__single')}}
    struct offset_nonnullable nn6 = { .buf = arg, .size = 0 };

    struct offset_nullable n1;
    struct offset_nullable n2 = {};
    // expected-error@+1{{possibly initializing 'n3.buf' of type 'int *__single __sized_by_or_null(size - 1)' (aka 'int *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
    struct offset_nullable n3 = { .buf = arg };
    struct offset_nullable n4 = { .size = 1, .buf = arg };
    struct offset_nullable n5 = { .size = 5, .buf = arg };
    // expected-error@+1{{possibly initializing 'n6.buf' of type 'int *__single __sized_by_or_null(size - 1)' (aka 'int *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
    struct offset_nullable n6 = { .buf = arg, .size = 0 };
}

