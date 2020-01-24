// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

template<typename T, typename U=void>
concept C = true;

int foo() {
  C auto a4 = 1;
  C<> auto a5 = 1;
  C<int> auto a6 = 1;
  const C auto &a7 = 1;
  const C<> auto &a8 = 1;
  const C<int> auto &a9 = 1;
  C decltype(auto) a10 = 1;
  C<> decltype(auto) a11 = 1;
  C<int> decltype(auto) a12 = 1;
  const C<> decltype(auto) &a13 = 1; // expected-error{{'decltype(auto)' cannot be combined with other type specifiers}}
  // expected-error@-1{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
  const C<int> decltype(auto) &a14 = 1; // expected-error{{'decltype(auto)' cannot be combined with other type specifiers}}
  // expected-error@-1{{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
  C a15 = 1;
  // expected-error@-1{{expected 'auto' or 'decltype(auto)' after concept name}}
  C decltype a19 = 1;
  // expected-error@-1{{expected '('}}
  C decltype(1) a20 = 1;
  // expected-error@-1{{expected 'auto' or 'decltype(auto)' after concept name}}
}