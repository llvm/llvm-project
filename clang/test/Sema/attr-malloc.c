// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -emit-llvm -o %t %s

#include <stddef.h>

// Declare malloc here explicitly so we don't depend on system headers.
void * malloc(size_t) __attribute((malloc));

int no_vars __attribute((malloc)); // expected-warning {{attribute only applies to functions}}

void  returns_void  (void) __attribute((malloc)); // expected-warning {{attribute only applies to return values that are pointers}}
int   returns_int   (void) __attribute((malloc)); // expected-warning {{attribute only applies to return values that are pointers}}
int * returns_intptr(void) __attribute((malloc)); // no-warning
typedef int * iptr;
iptr  returns_iptr  (void) __attribute((malloc)); // no-warning

__attribute((malloc)) void *(*f)(void); //  expected-warning{{attribute only applies to functions}}
__attribute((malloc)) int (*g)(void); // expected-warning{{attribute only applies to functions}}

__attribute((malloc))
void * xalloc(unsigned n) { return malloc(n); } // no-warning
// RUN: grep 'define .*noalias .* @xalloc(' %t %t

#define malloc_like __attribute((__malloc__))
void * xalloc2(unsigned) malloc_like;
void * xalloc2(unsigned n) { return malloc(n); }
// RUN: grep 'define .*noalias .* @xalloc2(' %t %t

