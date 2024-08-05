// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library  -finclude-default-header -verify %s

// Some things that don't work!

// Case 1: Both types must be complete!
struct Defined {
  int X;
};


struct Undefined; // expected-note {{forward declaration of 'Undefined'}}

_Static_assert(__is_scalarized_layout_compatible(Undefined, Defined), ""); // expected-error{{incomplete type 'Undefined' where a complete type is required}}

// Case 2: No variable length arrays!

void fn(int X) {
  // expected-error@#vla {{variable length arrays are not supported for the current target}}
  // expected-error@#vla {{variable length arrays are not supported in '__is_scalarized_layout_compatible'}}
  // expected-error@#vla {{static assertion failed due to requirement '__is_scalarized_layout_compatible(int[4], int[X])'}}
  // expected-warning@#vla {{variable length arrays in C++ are a Clang extension}}
  _Static_assert(__is_scalarized_layout_compatible(int[4], int[X]), ""); // #vla
}
