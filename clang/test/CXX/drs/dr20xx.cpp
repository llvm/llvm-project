// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors \
// RUN:            -Wno-variadic-macros -Wno-c11-extensions
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// expected-no-diagnostics

#if __cplusplus < 201103L
#define static_assert(...) _Static_assert(__VA_ARGS__)
#endif

namespace dr2094 { // dr2094: 5.0
  struct A { int n; };
  struct B { volatile int n; };
  static_assert(__is_trivially_copyable(volatile int), "");
  static_assert(__is_trivially_copyable(const volatile int), "");
  static_assert(__is_trivially_copyable(const volatile int[]), "");
  static_assert(__is_trivially_copyable(A), "");
  static_assert(__is_trivially_copyable(volatile A), "");
  static_assert(__is_trivially_copyable(const volatile A), "");
  static_assert(__is_trivially_copyable(const volatile A[]), "");
  static_assert(__is_trivially_copyable(B), "");

  static_assert(__is_trivially_constructible(A, A const&), "");
  static_assert(__is_trivially_constructible(B, B const&), "");

  static_assert(__is_trivially_assignable(A, const A&), "");
  static_assert(__is_trivially_assignable(B, const B&), "");
}
