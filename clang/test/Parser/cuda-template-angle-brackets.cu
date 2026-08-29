// RUN: %clang_cc1 -fsyntax-only -verify %s

template <template <template <int>>> struct S; // expected-error 2 {{template template parameter requires 'class' or 'typename' after the parameter list}}
