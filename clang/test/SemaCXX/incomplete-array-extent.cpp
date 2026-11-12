// RUN: %clang_cc1 -verify -std=c++14 %s

auto f() {
  return __array_extent(int, ); // expected-error {{expected expression}}
}
