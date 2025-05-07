

// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void outputs(void) {
  char * __bidi_indexable c_b_i;
	__asm__ volatile ("xyz %0" : "=r" (c_b_i));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
  char * __indexable c_i;
	__asm__ volatile ("xyz %0" : "=r" (c_i));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
  int n;
  char * __counted_by(n) c_b;
  __asm__ volatile ("xyz %0" : "=r" (c_b));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}

  char * __single s;
	__asm__ volatile ("xyz %0" : "=r" (s));

  char * r;
	__asm__ volatile ("xyz %0" : "=r" (r));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
}

void output_count(void) {
  int n;
  char * __counted_by(n) c_b;
  __asm__ volatile ("xyz %0" : "=r" (n));
  // expected-error@-1{{external count of a pointer cannot be used with inline assembly}}
}

struct Foo {
  int n;
  char * __counted_by(n) c_b;
};

void output_field(void) {
  struct Foo f;
  __asm__ volatile ("xyz %0" : "=r" (f.c_b));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
  __asm__ volatile ("xyz %0" : "=r" (f.n));
  // expected-error@-1{{external count of a pointer cannot be used with inline assembly}}
}

void output_ended_by(  char * end,  char * __ended_by(end) c_e_b) {
  __asm__ volatile ("xyz %0" : "=r" (c_e_b));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
}

void output_end(  char * end,  char * __ended_by(end) c_e_b) {
  __asm__ volatile ("xyz %0" : "=r" (end));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
}

void input_ended_by(  char * end,  char * __ended_by(end) c_e_b) {
  __asm__ volatile ("xyz %0" :: "r" (c_e_b));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}

  __asm__ volatile ("xyz %0" :: "r" (end));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
}

void inputs(unsigned long long temp) {
  unsigned foo;

  unsigned * __bidi_indexable u_b_i;
  __asm__ volatile ("xyz %1,%0"
    : "=r" (foo)
    : "r" (u_b_i));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}

  unsigned * __indexable u_i;
  __asm__ volatile ("xyz %1,%0"
    : "=r" (foo)
    : "r" (u_i));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}

  int n;
  unsigned * __counted_by(n) u_c_b;
  __asm__ volatile ("xyz %1,%0"
    : "=r" (foo)
    : "r" (u_c_b));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}

  __asm__ volatile ("xyz %1,%0"
                    : "=r" (foo)
                    : "r" (n));
  // expected-error@-1{{external count of a pointer cannot be used with inline assembly}}

  unsigned * __single u_s;
  __asm__ volatile ("xyz %1,%0"
    : "=r" (foo)
    : "r" (u_s));

  unsigned * u_r;
  __asm__ volatile ("xyz %1,%0"
    : "=r" (foo)
    : "r" (u_r));
  // expected-error@-1{{pointer with bounds cannot be used with inline assembly}}
}
