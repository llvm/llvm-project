// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// expected-no-diagnostics

template<typename T>
class X0 {
  friend T;
};

class X1 { };
enum E1 { };
X0<X1> x0a;
X0<X1 *> x0b;
X0<int> x0c;
X0<E1> x0d;
