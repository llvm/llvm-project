// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s

b;([c,ac])(){}
// expected-error@-1 2 {{a type specifier is required for all declarations}}
// expected-error@-2 {{structured binding declaration cannot be declared with parentheses}}
// expected-error@-3 {{expected expression}}
// expected-error@-4 {{expected ';' after top level declarator}}

([d])(){d}
// expected-error@-1 {{a type specifier is required for all declarations}}
// expected-error@-2 {{structured binding declaration cannot be declared with parentheses}}
// expected-error@-3 {{expected expression}}
// expected-error@-4 {{expected ';' after top level declarator}}

auto ([a])() {}
// expected-error@-1 {{structured binding declaration cannot be declared with parentheses}}
// expected-error@-2 {{expected expression}}
// expected-error@-3 {{expected ';' after top level declarator}}

auto (([x, y, z]))() {}
// expected-error@-1 {{structured binding declaration cannot be declared with parentheses}}
// expected-error@-2 {{expected expression}}
// expected-error@-3 {{expected ';' after top level declarator}}

struct S {
  ([m, n])() {}
  // expected-error@-1 {{structured binding declaration not permitted in this context}}
  // expected-error@-2 {{expected ';' at end of declaration list}}
};
