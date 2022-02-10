// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -std=c11 -isystem %S/Inputs
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wnull-pointer-subtraction -std=c11 -isystem %S/Inputs
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -std=c11 -isystem %S/Inputs -Wsystem-headers -DSYSTEM_WARNINGS
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wnull-pointer-subtraction -std=c11 -isystem %S/Inputs -Wsystem-headers -DSYSTEM_WARNINGS

#include <pointer-subtraction.h>

void a(void) {
  char *f = (char *)0;
  f = (char *)((char *)0 - f);         // expected-warning {{performing pointer subtraction with a null pointer has undefined behavior}}
  f = (char *)(f - (char *)0);         // expected-warning {{performing pointer subtraction with a null pointer has undefined behavior}}
  f = (char *)((char *)0 - (char *)0); // expected-warning {{performing pointer subtraction with a null pointer has undefined behavior}} expected-warning {{performing pointer subtraction with a null pointer has undefined behavior}}

#ifndef SYSTEM_WARNINGS
  SYSTEM_MACRO(f);
#else
  SYSTEM_MACRO(f); // expected-warning {{performing pointer subtraction with a null pointer has undefined behavior}}
#endif
}
