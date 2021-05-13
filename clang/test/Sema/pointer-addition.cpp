// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -std=c++11

void a() {
  char *f = (char*)0;
  f = (char*)((char*)0 - f); // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior}}
  f = (char*)(f - (char*)0); // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior}}
  f = (char*)((char*)0 - (char*)0); // valid in C++
}
