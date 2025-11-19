// RUN: %clang_cc1 -fsyntax-only -verify -Wno-everything %s

struct S {
  enum e1 {} // expected-error {{use of empty enum}} expected-error {{expected ';' after enum}}
  enum e2 {} // expected-error {{use of empty enum}}
}; // expected-error {{expected member name or ';' after declaration specifiers}}
