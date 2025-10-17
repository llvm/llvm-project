// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -emit-llvm -o %t %s

#include <stddef.h>

typedef struct {
  void *ptr;
  size_t n;
} sized_ptr;
sized_ptr  returns_sized_ptr  (void) __attribute((malloc_span)); // no-warning

// The first struct field must be pointer and the second must be an integer.
// Check the possible ways to violate it.
typedef struct {
  size_t n;
  void *ptr;
} invalid_span1;
invalid_span1  returns_non_std_span1  (void) __attribute((malloc_span)); // expected-warning {{attribute only applies to return values that are span-like structures}}

typedef struct {
  void *ptr;
  void *ptr2;
} invalid_span2;
invalid_span2  returns_non_std_span2  (void) __attribute((malloc_span)); // expected-warning {{attribute only applies to return values that are span-like structures}}

typedef struct {
  void *ptr;
  size_t n;
  size_t n2;
} invalid_span3;
invalid_span3  returns_non_std_span3  (void) __attribute((malloc_span)); // expected-warning {{attribute only applies to return values that are span-like structures}}
