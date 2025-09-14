// RUN: %clang_cc1 -std=c23 %s -verify

// Validate that the attribute works in C.
static_assert(!__has_c_attribute(assume));
static_assert(__has_c_attribute(clang::assume));
static_assert(__has_attribute(assume));

void test(int n) {
  // Smoke test.
  __attribute__((assume(true)));
  [[clang::assume(true)]];

  // Test diagnostics
  __attribute__((assume));    // expected-error {{'assume' attribute takes one argument}}
  __attribute__((assume()));  // expected-error {{expected expression}}
  [[clang::assume]];          // expected-error {{'assume' attribute takes one argument}}
  [[clang::assume()]];        // expected-error {{expected expression}}

  __attribute__((assume(n++))); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
  [[clang::assume(n++)]];       // expected-warning {{assumption is ignored because it contains (potential) side-effects}}

  [[clang::assume(true)]] int x;       // expected-error {{'clang::assume' attribute cannot be applied to a declaration}}
  __attribute__((assume(true))) int y; // expected-error {{'assume' attribute cannot be applied to a declaration}}
}

