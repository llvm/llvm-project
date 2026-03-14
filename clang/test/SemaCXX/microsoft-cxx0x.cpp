// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wc++11-narrowing -Wmicrosoft -verify -fms-extensions -std=c++11
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wc++11-narrowing -Wmicrosoft -verify -fms-extensions -std=c++11 -fms-compatibility -DMS_COMPAT


struct A {
     unsigned int a;
};
int b = 3;
A var = {  b }; // expected-warning {{ cannot be narrowed }} expected-note {{insert an explicit cast to silence this issue}}


namespace PR13433 {
  struct S;
  S make();

  template<typename F> auto x(F f) -> decltype(f(make()));
#ifndef MS_COMPAT
// expected-error@-2{{calling 'make' with incomplete return type 'S'}}
// expected-note@-5{{'make' declared here}}
// expected-note@-7{{forward declaration of 'PR13433::S'}}
#endif
}
