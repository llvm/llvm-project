

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-bounds-safety-attributes -x c %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-bounds-safety-attributes -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-bounds-safety-attributes -x objective-c %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-bounds-safety-attributes -x objective-c++ %s

#include <ptrcheck.h>

void foo(int *__counted_by(len + 1) buf, int len);
// expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
void bar(int *__counted_by(len) *buf __counted_by(len + 2), int len);
// expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
void baz(int *__counted_by(len) *__counted_by(len+2) buf, int len);

void byte_foo(int *__sized_by(len + 1) buf, int len);
// expected-error@+1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}
void byte_bar(int *__sized_by(len) *buf __sized_by(len + 2), int len);
// expected-error@+1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}
void byte_baz(int *__sized_by(len) *__sized_by(len+2) buf, int len);
void count_vla(int len, int buf[len + 1]);

void frob(int *__sized_by(len) buf, int len) {
    // expected-error-re@+1{{__typeof__ on an expression of type 'int *{{.*}}__sized_by(len)' (aka 'int *{{.*}}') is not yet supported}}
    __typeof__(buf) buf2;
}

void nicate(int *__sized_by(0) buf) {
    __typeof__(buf) buf2; // OK
}

void foo_or_null(int *__counted_by_or_null(len + 1) buf, int len);
// expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void bar_or_null(int *__counted_by_or_null(len) *buf __counted_by_or_null(len + 2), int len);
// expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void baz_or_null(int *__counted_by_or_null(len) *__counted_by_or_null(len+2) buf, int len);

void byte_foo_or_null(int *__sized_by_or_null(len + 1) buf, int len);
// expected-error@+1{{'__sized_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void byte_bar_or_null(int *__sized_by_or_null(len) *buf __sized_by_or_null(len + 2), int len);
// expected-error@+1{{'__sized_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void byte_baz_or_null(int *__sized_by_or_null(len) *__sized_by_or_null(len+2) buf, int len);
void count_vla_or_null(int len, int buf[len + 1]);

void frob_or_null(int *__sized_by_or_null(len) buf, int len) {
    // expected-error-re@+1{{__typeof__ on an expression of type 'int *{{.*}}__sized_by_or_null(len)' (aka 'int *{{.*}}') is not yet supported}}
    __typeof__(buf) buf2;
}

void nicate_or_null(int *__sized_by_or_null(0) buf) {
    __typeof__(buf) buf2; // OK
}
