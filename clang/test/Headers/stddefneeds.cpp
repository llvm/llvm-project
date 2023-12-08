// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Wsentinel -std=c++11 %s

ptrdiff_t p0; // expected-error{{unknown}}
size_t s0; // expected-error{{unknown}}
void* v0 = NULL; // expected-error{{undeclared}}
wint_t w0; // expected-error{{unknown}}
max_align_t m0; // expected-error{{unknown}}

#define __need_ptrdiff_t
#include <stddef.h>

ptrdiff_t p1;
size_t s1; // expected-error{{unknown}}
void* v1 = NULL; // expected-error{{undeclared}}
wint_t w1; // expected-error{{unknown}}
max_align_t m1; // expected-error{{unknown}}

#define __need_size_t
#include <stddef.h>

ptrdiff_t p2;
size_t s2;
void* v2 = NULL; // expected-error{{undeclared}}
wint_t w2; // expected-error{{unknown}}
max_align_t m2; // expected-error{{unknown}}

#define __need_NULL
#include <stddef.h>

ptrdiff_t p3;
size_t s3;
void* v3 = NULL;
wint_t w3; // expected-error{{unknown}}
max_align_t m3; // expected-error{{unknown}}

// Shouldn't bring in wint_t by default:
#include <stddef.h>

ptrdiff_t p4;
size_t s4;
void* v4 = NULL;
wint_t w4; // expected-error{{unknown}}
max_align_t m4;

#define __need_wint_t
#include <stddef.h>

ptrdiff_t p5;
size_t s5;
void* v5 = NULL;
wint_t w5;
max_align_t m5;


// linux/stddef.h does something like this for cpp files:
#undef NULL
#define NULL 0

// glibc (and other) headers then define __need_NULL and rely on stddef.h
// to redefine NULL to the correct value again.
#define __need_NULL
#include <stddef.h>

// gtk headers then use __attribute__((sentinel)), which doesn't work if NULL
// is 0.
void f(const char* c, ...) __attribute__((sentinel));
void g() {
  f("", NULL);  // Shouldn't warn.
}
