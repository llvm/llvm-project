// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S {};

union U {
   S x;
   float y;
};

void f() {
   new U{0,.y=1};
  // expected-warning@-1 {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
  // expected-note@-2 {{first non-designated initializer is here}}
  // expected-error@-3 {{initializer for aggregate with no elements requires explicit braces}}
}
