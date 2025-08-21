// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Wsentinel -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=old,expected -Wsentinel -std=c++98 %s

ptrdiff_t p0; // expected-error{{unknown}}
size_t s0; // expected-error{{unknown}}
void* v0 = NULL; // expected-error{{undeclared}}
wint_t w0; // expected-error{{unknown}}
max_align_t m0; // expected-error{{unknown}}
nullptr_t n0; // expected-error {{unknown}}

#define __need_ptrdiff_t
#include <stddef.h>

ptrdiff_t p1;
size_t s1; // expected-error{{unknown}}
void* v1 = NULL; // expected-error{{undeclared}}
wint_t w1; // expected-error{{unknown}}
max_align_t m1; // expected-error{{unknown}}
nullptr_t n1; // expected-error{{unknown}}

#define __need_size_t
#include <stddef.h>

ptrdiff_t p2;
size_t s2;
void* v2 = NULL; // expected-error{{undeclared}}
wint_t w2; // expected-error{{unknown}}
max_align_t m2; // expected-error{{unknown}}
nullptr_t n2; // expected-error{{unknown}}

#define __need_nullptr_t
#include <stddef.h>
ptrdiff_t p6;
size_t s6;
void* v6 = NULL; // expected-error{{undeclared}}
wint_t w6; // expected-error{{unknown}}
max_align_t m6; // expected-error{{unknown}}
nullptr_t n6; // old-error{{unknown}}

#define __need_NULL
#include <stddef.h>

ptrdiff_t p3;
size_t s3;
void* v3 = NULL;
wint_t w3; // expected-error{{unknown}}
max_align_t m3; // expected-error{{unknown}}
nullptr_t n3; // old-error{{unknown}}

#define __need_max_align_t
#include <stddef.h>
ptrdiff_t p7;
size_t s7;
void* v7 = NULL;
wint_t w7; // expected-error{{unknown}}
max_align_t m7;
nullptr_t n7; // old-error{{unknown}}

// Shouldn't bring in wint_t by default:
#include <stddef.h>

ptrdiff_t p4;
size_t s4;
void* v4 = NULL;
wint_t w4; // expected-error{{unknown}}
max_align_t m4;
nullptr_t n4; // old-error{{unknown}}

#define __need_wint_t
#include <stddef.h>

ptrdiff_t p5;
size_t s5;
void* v5 = NULL;
wint_t w5;
max_align_t m5;
nullptr_t n5; // old-error{{unknown}}

// linux/stddef.h does something like this for cpp files:
#undef NULL
#define NULL 0

// Including stddef.h again shouldn't redefine NULL
#include <stddef.h>

// gtk headers then use __attribute__((sentinel)), which doesn't work if NULL
// is 0.
void f(const char* c, ...) __attribute__((sentinel)); // expected-note{{function has been explicitly marked sentinel here}}
void g() {
  f("", NULL); // expected-warning{{missing sentinel in function call}}
}

// glibc (and other) headers then define __need_NULL and rely on stddef.h
// to redefine NULL to the correct value again.
#define __need_NULL
#include <stddef.h>

void h() {
  f("", NULL);  // Shouldn't warn.
}
