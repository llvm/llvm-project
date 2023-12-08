// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++1y -verify %s

template<typename T> struct S { typedef int type; };

template<typename T> void f() {
  auto x = [] { return 0; } ();
  // FIXME: We should be able to produce a 'missing typename' diagnostic here.
  S<decltype(x)>::type n; // expected-error 2{{}}
}

#if __cplusplus > 201103L
template<typename T> void g() {
  auto x = [] () -> auto { return 0; } ();
  S<decltype(x)>::type n; // expected-error 2{{}}
}
#endif
