// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// Don't crash (PR25181).

template <typename T> class Foo { // expected-note {{template parameter is declared here}}
  template <typename T> // expected-error {{declaration of 'T' shadows template parameter}}
  void Foo<T>::method(T *) const throw() {} // expected-error {{nested name specifier 'Foo<T>::' for declaration does not refer into a class, class template or class template partial specialization}}
};
