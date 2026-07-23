// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify -std=c++11 %s

typedef __SIZE_TYPE__ size_t;

void *operator new(size_t); // #new_decl

struct Tag {};

void f() {
  int *p = new (Tag{}) int[4];
  // expected-error@-1 {{no matching function for call to 'operator new'}}
  // expected-note@#new_decl {{candidate function not viable: requires 1 argument, but 2 were provided}}
  (void)p;
}
