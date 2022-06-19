// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -frounding-math -verify %s

template <class b> b::a() {}  // expected-error {{nested name specifier}}
