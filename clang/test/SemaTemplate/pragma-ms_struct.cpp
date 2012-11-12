// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-apple-osx10.7.0 %s
// expected-no-diagnostics

#pragma ms_struct on

// <rdar://problem/10791194>
template<int x> struct foo {
  long long a;
  int b;
};
extern int arr[sizeof(foo<0>) == 16 ? 1 : -1];
