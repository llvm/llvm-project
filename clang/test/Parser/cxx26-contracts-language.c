// RUN: %clang_cc1 -std=c17 -fcontracts -fsyntax-only -verify %s

// Contracts are a C++ feature. Enabling the parser option in C mode must not
// turn its spellings into keywords or contextual keywords.

// expected-no-diagnostics

int contract_assert;
int pre;
int post;

void use_contract_spellings(void) {
  ++contract_assert;
  ++pre;
  ++post;
}
