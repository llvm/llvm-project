// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

namespace type
{
  template<typename T>
  concept C1 = true;

  template<C1 T, C1 U = int>
  using A = T[10];

  using a = A<int>;

  namespace ns {
    template<typename T, int a = 0>
    concept C2 = true;
  }

  template<ns::C2 T1, ::type::ns::C2 T2> requires (sizeof(T1) <= sizeof(T2))
  struct B { };

  using b = B<int, int>;

  template<ns::C2... T1>
  struct C { };

  using c1 = C<char, char, char>;
  using c2 = C<char, char, char, char>;
}

namespace non_type
{
  template<int v>
  concept C1 = true;

  template<C1 v, C1 u = 0> // expected-error{{expected a type}} // expected-note{{declared here}}
  // expected-error@-1 2{{concept named in type constraint is not a type concept}}
  // expected-error@-2 {{expected ',' or '>' in template-parameter-list}}
  int A = v; // expected-error{{'v' does not refer to a value}}
}

namespace temp
{
  template<typename>
  struct test1 { }; // expected-note{{template is declared here}}

  template<template<typename> typename T>
  concept C1 = true;

  template<C1 TT, C1 UU = test1> // expected-error{{use of class template 'test1' requires template arguments}}
  // expected-error@-1 2{{concept named in type constraint is not a type concept}}
  using A = TT<int>; // expected-error{{expected ';' after alias declaration}}
}