// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++20 -fms-extensions -Wno-microsoft %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++23 -fms-extensions -Wno-microsoft %s

template <class T, class U>
struct Same {
  static constexpr auto value = __is_same(T, U);
};

template <class T>
struct __is_trivially_equality_comparable { // expected-warning{{keyword '__is_trivially_equality_comparable' will be made available as an identifier for the remainder of the translation unit}}
  using type = T;
};

using A = Same<__is_trivially_equality_comparable<int>::type, __is_trivially_equality_comparable<int>::type>;
