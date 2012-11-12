// RUN: %clang_cc1 %s -std=c++11 -fsyntax-only -verify -pedantic

# 1 "/usr/include/string.h" 1 3 4
extern "C" {
  typedef decltype(sizeof(int)) size_t;
  extern size_t strlen(const char *p);
}

# 10 "SemaCXX/constexpr-strlen.cpp" 2
constexpr int n = __builtin_strlen("hello"); // ok
constexpr int m = strlen("hello"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strlen' cannot be used in a constant expression}}

// Make sure we can evaluate a call to strlen.
int arr[3]; // expected-note {{here}}
int k = arr[strlen("hello")]; // expected-warning {{array index 5}}
