// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<class> class Foo {
  template<class UBar // expected-error {{expected ';' after class}}
                      // expected-note@-1 {{'UBar' declared here}}
  void foo1(); // expected-error {{a non-type template parameter cannot have type 'class UBar'}}
               // expected-error@-1 {{expected ',' or '>' in template-parameter-list}}
               // expected-warning@-2 {{declaration does not declare anything}}
};

Foo<int>::UBar g1; // expected-error {{no type named 'UBar' in 'Foo<int>'}}

class C0 {
public:
  template<typename T0, typename T1 = T0 // missing closing angle bracket
  struct S0 {}; // expected-error {{'S0' cannot be defined in a type specifier}}
                // expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
                // expected-error@-2 {{expected ',' or '>' in template-parameter-list}}
                // expected-warning@-3 {{declaration does not declare anything}}
  C0() : m(new S0<int>) {} // expected-error {{expected '(' for function-style cast or type construction}}
                           // expected-error@-1 {{expected expression}}
  S0<int> *m; // expected-error {{expected member name or ';' after declaration specifiers}}
};
