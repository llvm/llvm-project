// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected %s

namespace cwg2917 { // cwg2917: 20 open 2024-07-30
template <typename>
class Foo;

template<class ...> // cxx98-error {{variadic templates are a C++11 extension}}
struct C {
  struct Nested { };
};

struct S {
  template <typename>
  friend class Foo, int; // expected-error {{a friend declaration that befriends a template must contain exactly one type-specifier}}

  template <typename ...Ts> // cxx98-error {{variadic templates are a C++11 extension}}
  friend class C<Ts>::Nested...; // expected-error {{friend declaration expands pack 'Ts' that is declared it its own template parameter list}}
};
} // namespace cwg2917
