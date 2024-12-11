

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <stddef.h>
#include <ptrcheck.h>

void param_with_count(int *__counted_by(len - 2) buf, size_t len);

void param_with_count_size(int *__counted_by(len * size) buf, size_t len, size_t size);

void *__sized_by(size * len) return_count_size(size_t len, size_t size);

void *__sized_by(size * len) *return_count_size_ptr(size_t len, size_t size);
// expected-error@-1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}

void *__sized_by(size * *len) return_inout_count_size_ptr(size_t *len, size_t size);
// expected-error@-1{{invalid argument expression to bounds attribute}}

void *__sized_by(*len) return_inout_count(size_t *len);

void *__sized_by(*len + 1) return_inout_count_1(size_t *len);
// expected-error@-1{{invalid argument expression to bounds attribute}}

void *__sized_by_or_null(*len + 1) return_inout_count_2(size_t *len);
// expected-error@-1{{invalid argument expression to bounds attribute}}

int *__counted_by_or_null(*len + 1) return_inout_count_3(size_t *len);
// expected-error@-1{{invalid argument expression to bounds attribute}}

void *__sized_by(*len) *return_inout_count_ptr(size_t *len);
// expected-error@-1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}

void *__sized_by(*len + 1) *return_inout_count_1_ptr(size_t *len);
// expected-error@-1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}

void inout_buf(int *__counted_by(len * size) *buf, size_t len, size_t size);

void inout_buf_count(int *__counted_by(*len + 1) *buf, size_t *len);
// expected-error@-1{{invalid argument expression to bounds attribute}}

void inout_count(int *__counted_by(*len - 2) buf, size_t *len);
// expected-error@-1{{invalid argument expression to bounds attribute}}

