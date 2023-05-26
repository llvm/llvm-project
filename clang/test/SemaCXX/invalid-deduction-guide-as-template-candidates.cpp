// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

template <class T> class Foo {}; // expected-note {{candidate template ignored: couldn't infer template argument 'T'}} \
                                 // expected-note {{candidate function template not viable: requires 1 argument, but 0 were provided}}
Foo(); // expected-error {{deduction guide declaration without trailing return type}}
Foo vs; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'Foo'}}
