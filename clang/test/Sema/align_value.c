// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef double * __attribute__((align_value(64))) aligned_double;

void foo(aligned_double x, double * y __attribute__((align_value(32)))) { };

// expected-error@+1 {{requested alignment is not a power of 2}}
typedef double * __attribute__((align_value(63))) aligned_double1;

// expected-error@+1 {{requested alignment is not a power of 2}}
typedef double * __attribute__((align_value(-2))) aligned_double2;

// expected-error@+1 {{attribute takes one argument}}
typedef double * __attribute__((align_value(63, 4))) aligned_double3;

// expected-error@+1 {{attribute takes one argument}}
typedef double * __attribute__((align_value())) aligned_double3a;

// expected-error@+1 {{attribute takes one argument}}
typedef double * __attribute__((align_value)) aligned_double3b;

// expected-error@+1 {{'align_value' attribute requires integer constant}}
typedef double * __attribute__((align_value(4.5))) aligned_double4;

// expected-warning@+1 {{'align_value' attribute only applies to a pointer or reference ('int' is invalid)}}
typedef int __attribute__((align_value(32))) aligned_int;

typedef double * __attribute__((align_value(32*2))) aligned_double5;

// expected-warning@+1 {{'align_value' attribute only applies to variables and typedefs}}
void bar(void) __attribute__((align_value(32)));

