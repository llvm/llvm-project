// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1346 { // dr1346: 3.5
  auto a(1); // expected-error 0-1{{extension}}
  auto b(1, 2); // expected-error {{multiple expressions}} expected-error 0-1{{extension}}
#if __cplusplus >= 201103L
  auto c({}); // expected-error {{parenthesized initializer list}} expected-error {{cannot deduce}}
  auto d({1}); // expected-error {{parenthesized initializer list}} expected-error {{<initializer_list>}}
  auto e({1, 2}); // expected-error {{parenthesized initializer list}} expected-error {{<initializer_list>}}
#endif
  template<typename...Ts> void f(Ts ...ts) { // expected-error 0-1{{extension}}
    auto x(ts...); // expected-error {{empty}} expected-error 0-1{{extension}}
  }
  template void f(); // expected-note {{instantiation}}

#if __cplusplus >= 201103L
  void init_capture() {
    [a(1)] {} (); // expected-error 0-1{{extension}}
    [b(1, 2)] {} (); // expected-error {{multiple expressions}} expected-error 0-1{{extension}}
#if __cplusplus >= 201103L
    [c({})] {} (); // expected-error {{parenthesized initializer list}} expected-error {{cannot deduce}} expected-error 0-1{{extension}}
    [d({1})] {} (); // expected-error {{parenthesized initializer list}} expected-error {{<initializer_list>}} expected-error 0-1{{extension}}
    [e({1, 2})] {} (); // expected-error {{parenthesized initializer list}} expected-error {{<initializer_list>}} expected-error 0-1{{extension}}
#endif
  }
#endif
}
