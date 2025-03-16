// RUN: %clang_cc1 -fsyntax-only -Wpointer-bool-conversion %s
#include <stddef.h>
_Bool f() {
  int *p;
  if (p) {} // expected-warning {{implicit conversion of pointer to bool}}
  return (void *)0; // expected-warning {{implicit conversion of pointer to bool}}
}

_Bool g() {
  int *p = (void *)0;
  if (p == NULL) {} // no-warning
  return (void *)0 == NULL; // no-warning
}
