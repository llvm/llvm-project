

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-no-diagnostics

// This should get __sized_by_or_null(count * size).
void *my_calloc(unsigned count, unsigned size)
    __attribute__((alloc_size(1, 2)));

#define ALLOC(size) my_calloc(size, 1)

void test(void *__sized_by(*out_len) * out, unsigned *out_len) {
  *out = ALLOC(sizeof(int));
  *out_len = sizeof(int);
}
