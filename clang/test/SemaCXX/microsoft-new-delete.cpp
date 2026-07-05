// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify -std=c++11 %s

#include <stddef.h>

struct arbitrary_t {} arbitrary;
void *operator new(size_t size, arbitrary_t);

void f() {
  // Expect no error in MSVC compatibility mode
  int *p = new(arbitrary) int[4];
}

class noncopyable { noncopyable(const noncopyable&); } extern nc; // expected-note {{here}}
void *operator new[](size_t, noncopyable);
void *operator new(size_t, const noncopyable&);
void *q = new (nc) int[4]; // expected-error {{calling a private constructor}}

struct bitfield { int n : 3; } bf; // expected-note {{here}}
void *operator new[](size_t, int &); // expected-note {{passing argument to parameter here}}
void *operator new(size_t, const int &);
void *r = new (bf.n) int[4]; // expected-error {{non-const reference cannot bind to bit-field}}

struct base {};
struct derived : private base {} der; // expected-note {{here}}
void *operator new[](size_t, base &);
void *operator new(size_t, derived &);
void *s = new (der) int[4]; // expected-error {{private}}

struct explicit_ctor { explicit explicit_ctor(int); };
struct explicit_ctor_tag {} ect;
void *operator new[](size_t, explicit_ctor_tag, explicit_ctor);
void *operator new(size_t, explicit_ctor_tag, int);
void *t = new (ect, 0) int[4];
void *u = new (ect, {0}) int[4]; // expected-warning {{braces around scalar init}}
