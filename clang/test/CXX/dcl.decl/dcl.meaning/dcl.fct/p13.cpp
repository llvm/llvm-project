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

void f4i(int ... x); // expected-error{{type 'int' of function parameter pack does not contain any unexpanded parameter packs}}
void f4i0(int ...);

namespace array_type {
template<typename T>
void a(T[] ... x); // expected-error{{expected ')'}} expected-note{{to match this '('}}

template<typename T>
void b(T[] ...);

template<typename T>
void c(T ... []); // expected-error{{type 'T []' of function parameter pack does not contain any unexpanded parameter packs}}

template<typename T>
void d(T ... x[]); // expected-error{{type 'T []' of function parameter pack does not contain any unexpanded parameter packs}}

void ai(int[] ... x); // expected-error{{expected ')'}} expected-note{{to match this '('}}
void bi(int[] ...);
void ci(int ... []); // expected-error{{type 'int []' of function parameter pack does not contain any unexpanded parameter packs}}
void di(int ... x[]); // expected-error{{type 'int []' of function parameter pack does not contain any unexpanded parameter packs}}
}

void f5a(auto fp(int)->unk ...) {} // expected-error{{unknown type name 'unk'}}
void f5b(auto fp(int)->auto ...) {} // expected-error{{'auto' not allowed in function return type}}
void f5c(auto fp()->...) {} // expected-error{{expected a type}}

// FIXME: Expand for function and member pointer types.




