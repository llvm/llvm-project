// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X0 { };
struct X1 { };

template<typename T>
void f0() {
  const T *t = (const X0*)0; // expected-error{{cannot initialize a variable of type 'const X1 *' with an rvalue of type 'const X0 *'}}
}
template void f0<X1>(); // expected-note{{instantiation of}}
