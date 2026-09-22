// RUN: %clang_cc1 -verify %s -std=c++20

template<typename T> concept C = false; // expected-note {{because 'false' evaluated to false}}

void test() {
  _Atomic C<> auto &foo = 42; // expected-error {{deduced type 'int' does not satisfy 'C<>'}}
};
