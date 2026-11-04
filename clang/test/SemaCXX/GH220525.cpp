// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

namespace foo {
  template<typename T> struct S {};
}
template struct foo::S<foo>::bar; // expected-error {{unexpected namespace name 'foo': expected expression}}
