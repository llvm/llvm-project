// RUN: %clang_cc1 -std=c++17 -fopenmp -fsyntax-only -verify %s

// This file tests the custom parsing logic for the OpenMP 5.1 attribute
// syntax. It does not test actual OpenMP directive syntax, just the attribute
// parsing bits.

// FIXME: the diagnostic here is a bit unsatisfying. We handle the custom omp
// attribute parsing logic when parsing the attribute argument list, and we
// only process an attribute argument list when we see an open paren after the
// attribute name. So this means we never hit the omp-specific parsing and
// instead handle this through the usual Sema attribute handling in
// SemaDeclAttr.cpp, which diagnoses this as an unknown attribute.
[[omp::directive]]; // expected-warning {{unknown attribute 'directive' ignored}}
[[omp::sequence]]; // expected-warning {{unknown attribute 'sequence' ignored}}
[[omp::unknown]]; // expected-warning {{unknown attribute 'unknown' ignored}}

[[omp::directive()]]; // expected-error {{expected an OpenMP directive}}
[[omp::sequence()]]; // expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}

// Both sequence and directive require an argument list, test that we diagnose
// when the inner directive or sequence is missing its argument list.
[[omp::sequence(directive)]]; // expected-error {{expected '('}}
[[omp::sequence(sequence)]]; // expected-error {{expected '('}}
[[omp::sequence(omp::directive)]]; // expected-error {{expected '('}}
[[omp::sequence(omp::sequence)]]; // expected-error {{expected '('}}

// All of the diagnostics here come from the inner sequence and directive not
// being given an argument, but this tests that we can parse either with or
// without the 'omp::'.
[[omp::sequence(directive(), sequence())]]; // expected-error {{expected an OpenMP directive}} expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[omp::sequence(omp::directive(), sequence())]]; // expected-error {{expected an OpenMP directive}} expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[omp::sequence(directive(), omp::sequence())]]; // expected-error {{expected an OpenMP directive}} expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[omp::sequence(omp::directive(), omp::sequence())]]; // expected-error {{expected an OpenMP directive}} expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}

// Test that we properly diagnose missing parens within the inner arguments of
// a sequence attribute.
[[omp::sequence( // expected-note {{to match this '('}}
  directive(
)]]; // expected-error {{expected ')'}} expected-error {{expected an OpenMP directive}}
[[omp::sequence( // expected-note {{to match this '('}}
  sequence(
)]]; // expected-error {{expected ')'}} expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}

// Test that we properly handle the using attribute syntax.
[[using omp: directive()]]; // expected-error {{expected an OpenMP directive}}
[[using omp: sequence()]]; // expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[using omp: sequence(omp::directive())]]; // expected-error {{expected an OpenMP directive}}
[[using omp: sequence(directive())]]; // expected-error {{expected an OpenMP directive}}

// Test that we give a sensible error on an unknown attribute in the omp
// namespace that has an argument list.
[[omp::unknown()]]; // expected-warning {{unknown attribute 'unknown' ignored}}
[[using omp: unknown()]]; // expected-warning {{unknown attribute 'unknown' ignored}}

// Test that unknown arguments to the omp::sequence are rejected, regardless of
// what level they're at.
[[omp::sequence(unknown)]]; // expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[omp::sequence(sequence(unknown))]]; // expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[omp::sequence(omp::unknown)]]; // expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}
[[omp::sequence(sequence(omp::unknown))]]; // expected-error {{expected an OpenMP 'directive' or 'sequence' attribute argument}}

// FIXME: combining non-openmp attributes with openmp attributes has surprising
// results due to the replay of tokens. We properly parse the non-openmp
// attributes, but we also replay the OpenMP tokens. The attributes then get
// passed to the OpenMP parsing functions and it does not attach the attribute
// to the declaration statement AST node as you might expect. This means that
// the expected diagnostics are not issued. Thankfully, due to the positioning
// of OpenMP attributes and what they appertain to, this should not be a
// frequent issue (hopefully).
int x;
[[deprecated, omp::directive(threadprivate(x))]] int y; // FIXME-expected-note {{'y' has been explicitly marked deprecated here}}
[[omp::directive(threadprivate(x)), deprecated]] int z; // FIXME-expected-note {{'z' has been explicitly marked deprecated here}}
void test() {
  x = 1;
  y = 1; // FIXME-expected-warning {{warning: 'y' is deprecated}}
  z = 1; // FIXME-expected-warning {{warning: 'z' is deprecated}}
}
