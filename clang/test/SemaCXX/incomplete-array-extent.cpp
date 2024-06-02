// RUN: %clang_cc1 -verify -std=c++11 %s

auto f() { // expected-error {{'auto' return without trailing return type; deduced return types are a C++14 extension}}
  return __array_extent(int, ); // expected-error {{expected expression}}
}
