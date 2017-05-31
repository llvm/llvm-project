// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -ftemplate-depth 16 -fcxx-exceptions -fexceptions %s

template<typename T> T go(T a) noexcept(noexcept(go(a))); // \
// expected-error 16{{call to function 'go' that is neither visible}} \
// expected-note 16{{'go' should be declared prior to the call site}} \
// expected-error {{recursive template instantiation exceeded maximum depth of 16}}

void f() {
  int k = go(0); // \
  // expected-note {{in instantiation of exception specification for 'go<int>' requested here}}
}
