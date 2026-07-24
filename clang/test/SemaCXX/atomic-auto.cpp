// RUN: %clang_cc1 -verify -pedantic %s -std=c++20

template<typename T> concept C = false; // expected-note {{because 'false' evaluated to false}}

void test() {
  _Atomic C<> auto &foo = 42; // expected-warning {{'_Atomic' is a C11 extension}} \
                              // expected-error {{deduced type 'int' does not satisfy 'C<>'}}
};
