// RUN: %clang_cc1 -fobjc-exceptions -fexceptions -std=c++11 -fblocks -fsyntax-only -verify %s

template<typename...Types>
void f(Types ...values) {
  for (id x in values) { } // expected-error {{expression contains unexpanded parameter pack 'values'}}
  @synchronized(values) { // expected-error {{expression contains unexpanded parameter pack 'values'}}
    @throw values; // expected-error {{expression contains unexpanded parameter pack 'values'}}
  }
}
