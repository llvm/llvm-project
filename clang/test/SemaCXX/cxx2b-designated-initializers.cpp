// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify %s

namespace PR61118 {

union S {
  struct {
    int a;
  };
};

void f(int x, auto) {
  const S result { // expected-error {{field designator (null) does not refer to any field in type 'const S'}}
    .a = x
  };
}

void g(void) {
  f(0, 0); // expected-note {{in instantiation of function template specialization 'PR61118::f<int>' requested here}}
}

} // end namespace PR61118
