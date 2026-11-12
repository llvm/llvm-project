// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -verify %s -pedantic-errors -DPEDANTIC
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -verify %s -Wno-c++11-narrowing

namespace cce_narrowing {
  decltype(short{123456}) a;
#if PEDANTIC
  // expected-error@-2 {{cannot be narrowed}} expected-note@-2 {{cast}}
#endif

  template<typename T> int f(decltype(T{123456})); // expected-note {{cannot be narrowed}}
  int b = f<short>(0); // expected-error {{no match}}
}
