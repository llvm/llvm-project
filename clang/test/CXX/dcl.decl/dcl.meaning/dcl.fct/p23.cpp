// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify %s

// FIXME: This should be an error with -pedantic-errors.
template<> // expected-warning {{extraneous template parameter list in template specialization}}
void f(auto);

template<typename>
void f(auto);

template<typename T>
struct A {
  void g(auto);
};

template<typename T>
void A<T>::g(auto) { }

template<>
void A<int>::g(auto) { }

// FIXME: This should be an error with -pedantic-errors.
template<>
template<> // expected-warning {{extraneous template parameter list in template specialization}}
void A<long>::g(auto) { }
