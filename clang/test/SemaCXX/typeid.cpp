// RUN: %clang_cc1 -fsyntax-only -verify %s

void f()
{
  (void)typeid(int); // expected-error {{you need to include <typeinfo> before using the 'typeid' operator}}
}

namespace std {
  struct type_info { const char *name; };
}

void g()
{
  (void)typeid(int);
}

struct X; // expected-note 3{{forward declaration}}

void g1(X &x) {
  (void)typeid(X); // expected-error{{'typeid' of incomplete type 'X'}}
  (void)typeid(X&); // expected-error{{'typeid' of incomplete type 'X'}}
  (void)typeid(x); // expected-error{{'typeid' of incomplete type 'X'}}
}

void h(int i) {
  char V[i];
  typeid(V);        // expected-error{{'typeid' of variably modified type 'char[i]'}}
  typeid(char [i]); // expected-error{{'typeid' of variably modified type 'char[i]'}}
}

// expected-note@+1 {{read of object 'typeid(int).name' whose value is not known}}
constexpr const char *name = typeid(int).name; // expected-error {{constant expression}}
