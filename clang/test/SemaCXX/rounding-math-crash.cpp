// RUN: %clang_cc1 -fsyntax-only -frounding-math -verify %s

template <class b> b::a() {}  // expected-error {{nested name specifier}}
