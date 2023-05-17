
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -fopenmp %s

// FIXME: OpenMP should support capturing structured bindings
auto f() {
  int i[2] = {};
  auto [a, b] = i; // expected-note 2{{declared here}}
  return [=, &a] {
    // expected-error@-1 {{capturing a structured binding is not yet supported in OpenMP}}
    return a + b;
    // expected-error@-1 {{capturing a structured binding is not yet supported in OpenMP}}
  };
}
