// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

template<typename T, typename U=void>
concept C = true;

namespace ns {
  template<typename T, typename U=void>
  concept D = true;
}

int foo() {
  {ns::D auto a = 1;}
  {C auto a = 1;}
  {C<> auto a = 1;}
  {C<int> auto a = 1;}
  {ns::D<int> auto a = 1;}
  {const ns::D auto &a = 1;}
  {const C auto &a = 1;}
  {const C<> auto &a = 1;}
  {const C<int> auto &a = 1;}
  {const ns::D<int> auto &a = 1;}
  {C decltype(auto) a = 1;}
  {C<> decltype(auto) a = 1;}
  {C<int> decltype(auto) a = 1;}
  {const C<> decltype(auto) &a = 1;} // expected-error{{'decltype(auto)' cannot be combined with other type specifiers}}
  // expected-error@-1{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
  {const C<int> decltype(auto) &a = 1;} // expected-error{{'decltype(auto)' cannot be combined with other type specifiers}}
  // expected-error@-1{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
  {C a = 1;}
  // expected-error@-1{{expected 'auto' or 'decltype(auto)' after concept name}}
  {C decltype a19 = 1;}
  // expected-error@-1{{expected '('}}
  {C decltype(1) a20 = 1;}
  // expected-error@-1{{expected 'auto' or 'decltype(auto)' after concept name}}
}