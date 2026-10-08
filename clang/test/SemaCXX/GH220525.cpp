// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

namespace foo {
  template<typename T> struct S {}; // expected-note {{template parameter is declared here}}
}
template struct foo::S<foo>::bar; // expected-error {{template argument for template type parameter must be a type}} \
                                  // expected-error {{unexpected namespace name 'foo': expected expression}}
