// RUN: %clang_cc1 -std=c++1y -fsyntax-only -verify %s -DMAX=1234 -fconstexpr-steps=1234
// RUN: %clang_cc1 -std=c++1y -fsyntax-only -verify %s -DMAX=10 -fconstexpr-steps=10
// RUN: %clang -std=c++1y -fsyntax-only -Xclang -verify %s -DMAX=12345 -fconstexpr-steps=12345

// This takes a total of n + 4 steps according to our current rules:
//  - One for the compound-statement that is the function body
//  - One for the 'for' statement
//  - One for the 'int k = 0;' statement
//  - One for each of the n evaluations of the compound-statement in the 'for' body
//  - One for the 'return' statemnet
constexpr bool steps(int n) {
  for (int k = 0; k != n; ++k) {}
  return true; // expected-note {{step limit}}
}

static_assert(steps((MAX - 4)), ""); // ok
static_assert(steps((MAX - 3)), ""); // expected-error {{constant}} expected-note{{call}}
