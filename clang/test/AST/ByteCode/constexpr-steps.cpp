// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s -fconstexpr-steps=100


constexpr int foo() { // expected-error {{never produces a constant expression}}
  while (1) {} // expected-note 2{{constexpr evaluation hit maximum step limit}}
  return 0;
}
static_assert (foo() == 0, ""); // expected-error {{not an integral constant expression}} \
                                // expected-note {{in call to}}

