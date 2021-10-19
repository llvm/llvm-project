// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -std=c++11 -isystem %S/Inputs
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wnull-pointer-subtraction -std=c++11 -isystem %S/Inputs
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -std=c++11 -isystem %S/Inputs -Wsystem-headers -DSYSTEM_WARNINGS
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wnull-pointer-subtraction -std=c++11 -isystem %S/Inputs -Wsystem-headers -DSYSTEM_WARNINGS

#include <pointer-subtraction.h>

void a() {
  char *f = (char *)0;
  f = (char *)((char *)0 - f);         // expected-warning {{performing pointer subtraction with a null pointer may have undefined behavior}}
  f = (char *)(f - (char *)0);         // expected-warning {{performing pointer subtraction with a null pointer may have undefined behavior}}
  f = (char *)((char *)0 - (char *)0); // valid in C++

#ifndef SYSTEM_WARNINGS
  SYSTEM_MACRO(f);
#else
  SYSTEM_MACRO(f); // expected-warning {{performing pointer subtraction with a null pointer may have undefined behavior}}
#endif
}
