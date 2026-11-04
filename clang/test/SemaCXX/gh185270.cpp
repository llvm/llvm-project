// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

void foo()([] consteval -> int { return 0; }()); // expected-error {{illegal initializer (only variables can be initialized)}}
