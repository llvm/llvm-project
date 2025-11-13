// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -emit-llvm -o %t %s

typedef __SIZE_TYPE__ size_t;

typedef struct {
  void *ptr;
  size_t n;
} span;
span  returns_span  (void) __attribute((malloc_span)); // no-warning

// Try out a different field ordering.
typedef struct {
  size_t n;
  void *ptr;
} span2;
span2  returns_span2  (void) __attribute((malloc_span)); // no-warning

// Ensure that a warning is produced on malloc_span precondition violation.
typedef struct {
  void *ptr;
  void *ptr2;
} invalid_span1;
invalid_span1  returns_non_std_span1  (void) __attribute((malloc_span)); // expected-warning {{attribute only applies to functions that return span-like structures}}

typedef struct {
  void *ptr;
  size_t n;
  size_t n2;
} invalid_span2;
invalid_span2  returns_non_std_span2  (void) __attribute((malloc_span)); // expected-warning {{attribute only applies to functions that return span-like structures}}
