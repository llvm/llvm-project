// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -fexceptions -verify %s

// When it is part of a parameter-declaration-clause, the parameter
// pack is a function parameter pack.
template<typename ...Types>
void f0(Types ...args);

template<typename ...Types>
void f1(const Types &...args);

// [ Note: Otherwise, the parameter-declaration is part of a
// template-parameter-list and the parameter pack is a template
// parameter pack; see 14.1. -- end note ]
template<int ...N>
struct X0 { };

template<typename ...Types>
struct X1 {
  template<Types ...Values> struct Inner;
};

// A declarator-id or abstract-declarator containing an ellipsis shall
// only be used in a parameter-declaration.
int (...f2)(int); // expected-error{{only function and template parameters can be parameter packs}}

void f3() {
  int ...x; // expected-error{{only function and template parameters can be parameter packs}}
  if (int ...y = 17) { }  // expected-error{{only function and template parameters can be parameter packs}}

  for (int ...z = 0; z < 10; ++z) { }  // expected-error{{only function and template parameters can be parameter packs}}

  try {
  } catch (int ...e) { // expected-error{{only function and template parameters can be parameter packs}}
  }
}

template<typename ...Types>
struct X2 {
  Types ...members; // expected-error{{only function and template parameters can be parameter packs}} \
                    // expected-error{{data member type contains unexpanded parameter pack}}
};

// The type T of the declarator-id of the function parameter pack
// shall contain a template parameter pack; each template parameter
// pack in T is expanded by the function parameter pack.
template<typename T>
void f4(T ...args); // expected-error{{type 'T' of function parameter pack does not contain any unexpanded parameter packs}}

