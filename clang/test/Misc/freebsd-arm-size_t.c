// RUN: %clang_cc1 -triple arm-unknown-freebsd10.0 -verify %s
// expected-no-diagnostics

/* Define a size_t as expected for FreeBSD ARM */
typedef unsigned int size_t;

/* Declare a builtin function that uses size_t */
void *malloc(size_t);

