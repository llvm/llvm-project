// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
class X0 {
  friend T; // expected-warning{{non-class friend type 'T' is a C++11 extension}}
};

class X1 { };
enum E1 { };
X0<X1> x0a;
X0<X1 *> x0b;
X0<int> x0c;
X0<E1> x0d;

