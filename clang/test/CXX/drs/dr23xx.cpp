// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus <= 201103L
// expected-no-diagnostics
#endif

namespace dr2387 { // dr2387: 9
#if __cplusplus >= 201402L
  template<int> int a = 0;
  extern template int a<0>; // ok

  template<int> static int b = 0;
  extern template int b<0>; // expected-error {{internal linkage}}

  template<int> const int c = 0;
  extern template const int c<0>; // ok, has external linkage despite 'const'

  template<typename T> T d = 0;
  extern template int d<int>;
  extern template const int d<const int>;
#endif
}
