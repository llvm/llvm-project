// RUN: %clang_cc1 -fsyntax-only -verify %s

// GH183505: Ensure we don't crash when stripping parentheses from placeholder types.

void test_unresolved_lookup() {
  (int)(abc<>); // expected-error {{use of undeclared identifier 'abc'}}
  (int)((abc<>)); // expected-error {{use of undeclared identifier 'abc'}}
}
