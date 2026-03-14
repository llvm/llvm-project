// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -std=c++11 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic-errors -Werror -x c++ -std=c++11 %t
// RUN: %clang_cc1 -Wall -pedantic-errors -Werror -x c++ -std=c++1y %t

// This is a test of the code modification hints for C++1y-compatibility problems.

struct S {
  constexpr int &f(); // expected-warning {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const' to avoid a change in behavior}}
  int &f();
};
