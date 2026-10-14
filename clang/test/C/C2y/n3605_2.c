// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s

static_assert(1, _Generic(1, default: "Error Message"));  // expected-error{{expected string literal for diagnostic message in static_assert}}
