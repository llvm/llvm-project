// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++20 -fms-extensions -Wno-microsoft %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++23 -fms-extensions -Wno-microsoft %s

template <class T, class U>
struct Same {
  static constexpr auto value = __is_same(T, U);
};

template <class T>
struct __is_nothrow_convertible {
  // expected-warning@-1 {{using the name of the builtin '__is_nothrow_convertible' outside of a builtin invocation is deprecated}}
  using type = T;
};

using A = Same<
  __is_nothrow_convertible<int>::type,
  // expected-warning@-1 {{using the name of the builtin '__is_nothrow_convertible' outside of a builtin invocation is deprecated}}
  __is_nothrow_convertible<int>::type
  // expected-warning@-1 {{using the name of the builtin '__is_nothrow_convertible' outside of a builtin invocation is deprecated}}
>;

