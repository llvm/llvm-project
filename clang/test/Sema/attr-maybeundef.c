// RUN: %clang_cc1  -fsyntax-only -verify %s

// Decl annotations.
void f(int __attribute__((maybe_undef)) *a);
void (*fp)(int __attribute__((maybe_undef)) handle);
__attribute__((maybe_undef)) int i(); // expected-warning {{'maybe_undef' attribute only applies to parameters}}
int __attribute__((maybe_undef)) a; // expected-warning {{'maybe_undef' attribute only applies to parameters}}
int (* __attribute__((maybe_undef)) fpt)(char *); // expected-warning {{'maybe_undef' attribute only applies to parameters}}
void h(int *a __attribute__((maybe_undef("RandomString")))); // expected-error {{'maybe_undef' attribute takes no arguments}}

// Type annotations.
int __attribute__((maybe_undef)) ta; // expected-warning {{'maybe_undef' attribute only applies to parameters}}

// Typedefs.
typedef int callback(char *) __attribute__((maybe_undef)); // expected-warning {{'maybe_undef' attribute only applies to parameters}}
