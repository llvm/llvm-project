// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <class T>
concept C = true;

template <template <class> concept... CC> // expected-error {{concept template parameter is a C++2c extension}}
int foo(CC &) { return 42; } // expected-error {{expected 'auto' or 'decltype(auto)' after concept name}} \
                             // expected-note {{candidate template ignored: constraints not satisfied}}

static_assert(foo<C>(0) == 42, ""); // expected-error {{no matching function for call to 'foo'}}