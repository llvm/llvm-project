// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: This should really include <typeinfo>, but we don't have that yet.
namespace std {
  class type_info;
}

void f()
{
  (void)typeid(int);
  (void)typeid(0);
  (void)typeid 1; // expected-error {{expected '(' after 'typeid'}}
}
