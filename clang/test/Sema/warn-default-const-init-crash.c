// RUN: %clang_cc1 -fsyntax-only -verify %s

// This invalid code was causing a stack overflow, check that we issue
// reasonable diagnostics and not crash.
struct GH140887 {    // expected-note {{definition of 'struct GH140887' is not complete until the closing '}'}}
  struct GH140887 s; // expected-error {{field has incomplete type 'struct GH140887'}}
};

void gh140887() {
  struct GH140887 s;
}
