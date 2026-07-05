// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete; // expected-note {{here}}
};

template<typename ...T>
void f(T ...t) {
  auto x(t...); // expected-error {{is empty}} expected-error {{contains multiple expressions}}
  only<int> check = x;
}

void g() {
  f(); // expected-note {{here}}
  f(0);
  f(0, 1); // expected-note {{here}}
}


template<typename T>
bool h(T t) {
  auto a = t;
  decltype(a) b;
  a = a + b;

  auto p = new auto(t);

  only<double*> test = p; // expected-error {{conversion function from 'char *' to 'only<double *>'}}
  return p;
}

bool b = h('x'); // expected-note {{here}}

// PR 9276 - Make sure we check auto types deduce the same
// in the case of a dependent initializer
namespace PR9276 {
  template<typename T>
  void f() {
    auto i = T(), j = 0; // expected-error {{deduced as 'long' in declaration of 'i' and deduced as 'int' in declaration of 'j'}}
  }

  void g() {
    f<long>(); // expected-note {{here}}
    f<int>();
  }
}

namespace NoRepeatedDiagnostic {
  template<typename T>
  void f() {
    auto a = 0, b = 0.0, c = T(); // expected-error {{deduced as 'int' in declaration of 'a' and deduced as 'double' in declaration of 'b'}}
  }
  // We've already diagnosed an issue. No extra diagnostics is needed for these.
  template void f<int>();
  template void f<double>();
  template void f<char>();
}
