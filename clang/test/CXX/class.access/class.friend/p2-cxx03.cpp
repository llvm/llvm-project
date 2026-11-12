// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
template<typename T>
class X0 {
  friend T;
#if __cplusplus <= 199711L // C++03 or earlier modes
  // expected-warning@-2{{non-class friend type 'T' is a C++11 extension}}
#else
  // expected-no-diagnostics
#endif
};

class X1 { };
enum E1 { };
X0<X1> x0a;
X0<X1 *> x0b;
X0<int> x0c;
X0<E1> x0d;

