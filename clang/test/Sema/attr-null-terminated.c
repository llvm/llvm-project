// RUN: %clang_cc1 -fsyntax-only -verify %s

void good_ptr(__attribute__((null_terminated)) const int *p);
void good_arr(__attribute__((null_terminated)) const int p[]);
void good_char(__attribute__((null_terminated)) const char *s);

// Not a pointer or array type
void bad_int(__attribute__((null_terminated)) int x); // expected-warning {{'null_terminated' attribute only applies to parameters of pointer or array type}}

// Not a parameter
__attribute__((null_terminated)) int global;  // expected-warning {{'null_terminated' attribute only applies to parameters}}
struct S { __attribute__((null_terminated)) int *field; };  // expected-warning {{'null_terminated' attribute only applies to parameters}}

// Takes no arguments
void bad_args0(__attribute__((null_terminated("test"))) const int *p); // expected-error {{'null_terminated' attribute takes no arguments}}
void bad_args1(__attribute__((null_terminated(123))) const int *p); // expected-error {{'null_terminated' attribute takes no arguments}}
