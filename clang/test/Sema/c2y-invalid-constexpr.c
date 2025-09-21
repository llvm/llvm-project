// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s

// This was previously causing a stack overflow when checking the valid
// declaration of an invalid type. Ensure we issue reasonable diagnostics
// instead of crashing.
struct GH140887 { // expected-note {{definition of 'struct GH140887' is not complete until the closing '}'}}
  GH140887();     // expected-error {{must use 'struct' tag to refer to type 'GH140887'}} \
                     expected-error {{expected member name or ';' after declaration specifiers}} \
                     expected-error {{field has incomplete type 'struct GH140887'}}
};
constexpr struct GH140887 a; // expected-error {{constexpr variable 'a' must be initialized by a constant expression}}

