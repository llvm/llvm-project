// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s

// Some things that don't work!

// Case 1: Both types must be complete!
struct Defined {
  int X;
};


struct Undefined; // expected-note {{forward declaration of 'Undefined'}}

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Undefined, Defined), ""); // expected-error{{incomplete type 'Undefined' where a complete type is required}}

// Case 2: No variable length arrays!

void fn(int X) { // expected-note{{declared here}}
  // expected-error@#vla {{variable length arrays are not supported for the current target}}
  // expected-error@#vla {{variable length arrays are not supported in '__builtin_hlsl_is_scalarized_layout_compatible'}}
  // expected-error@#vla {{static assertion failed due to requirement '__builtin_hlsl_is_scalarized_layout_compatible(int[4], int[X])'}}
  // expected-warning@#vla {{variable length arrays in C++ are a Clang extension}}
  // expected-note@#vla{{function parameter 'X' with unknown value cannot be used in a constant expression}}
  _Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(int[4], int[X]), ""); // #vla
}

// Case 3: Make this always fail for unions.
// HLSL doesn't really support unions, and the places where scalarized layouts
// are valid is probably going to be really confusing for unions, so we should
// just make sure unions are never scalarized compatible with anything other
// than themselves.

union Wah {
  int OhNo;
  float NotAgain;
};

struct OneInt {
  int I;
};

struct OneFloat {
  float F;
};

struct HasUnion {
  int I;
  Wah W;
};

struct HasUnionSame {
  int I;
  Wah W;
};

struct HasUnionDifferent {
  Wah W;
  int I;
};

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(Wah, Wah), "Identical types are always compatible");
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(Wah, OneInt), "Unions are not compatible with anything else");
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(Wah, OneFloat), "Unions are not compatible with anything else");

_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(HasUnion, HasUnionSame), "");
_Static_assert(!__builtin_hlsl_is_scalarized_layout_compatible(HasUnion, HasUnionDifferent), "");
