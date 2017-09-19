// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -std=c++11

#include <stdint.h>

void f(intptr_t offset, int8_t b) {
  // A zero offset from a nullptr is OK.
  char *f = (char*)nullptr + 0;
  int *g = (int*)0 + 0;
  f = (char*)nullptr - 0;
  g = (int*)nullptr - 0;
  // adding other values is undefined.
  f = (char*)nullptr + offset; // expected-warning {{arithmetic on a null pointer treated as a cast from integer to pointer is a GNU extension}}
  // Cases that don't match the GNU inttoptr idiom get a different warning.
  f = (char*)0 - offset; // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior if the offset is nonzero}}
  g = (int*)0 + offset; // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior if the offset is nonzero}}
  f = (char*)0 + b; // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior if the offset is nonzero}}
}

// Value-dependent pointer arithmetic should not produce a nullptr warning.
template<char *P>
char* g(intptr_t offset) {
  return P + offset;
}

// Value-dependent offsets should not produce a nullptr warning.
template<intptr_t N>
char *h() {
  return (char*)nullptr + N;
}
