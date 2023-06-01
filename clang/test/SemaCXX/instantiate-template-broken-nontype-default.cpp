// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

constexpr Missing a = 0; // expected-error {{unknown type name 'Missing'}}

template < typename T, Missing b = a> // expected-error {{unknown type name 'Missing'}}
class Klass { // expected-note {{candidate template ignored: could not match 'Klass<T, b>' against 'int'}}
  Klass(T);   // expected-note {{candidate template ignored: substitution failure [with T = int, b = <recovery-expr>()]}}
};

Klass foo{5}; // no-crash \
                 expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'Klass'}}

