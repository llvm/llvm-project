// RUN: %clang_cc1 -verify %s

template<typename T> struct S {
  static int a, b;
};

template<typename T> int S<T>::a, S<T>::b; // expected-error {{can only declare a single entity}}

template<typename T> struct A { static A a; } A<T>::a; // expected-error {{expected ';' after struct}} \
                                                          expected-error {{use of undeclared identifier 'T'}}

template<typename T> struct B { } f(); // expected-error {{expected ';' after struct}} \
                                          expected-error {{a type specifier is required}}

template<typename T> struct C { } // expected-error {{expected ';' after struct}}

A<int> c;

struct D {
  template<typename T> static const int x = 0, f(); // expected-error {{can only declare a single entity}}

  template<typename T> static const int g(), y = 0; // expected-error {{can only declare a single entity}}
};
