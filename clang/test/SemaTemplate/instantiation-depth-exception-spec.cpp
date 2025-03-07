// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -ftemplate-depth=16 -fcxx-exceptions -fexceptions %s

template<int N> struct X { // \
// expected-error {{recursive template instantiation exceeded maximum depth of 16}} \
// expected-note {{use -ftemplate-depth}}
  static int go(int a) noexcept(noexcept(X<N+1>::go(a))); // \
// expected-note 9{{in instantiation of exception specification}} \
// expected-note {{skipping 7 context}}
};

void f() {
  int k = X<0>::go(0); // \
  // expected-note {{in instantiation of exception specification for 'go' requested here}}
}
