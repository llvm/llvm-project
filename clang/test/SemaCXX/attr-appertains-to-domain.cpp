// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

void reject_declaration_attribute_on_statement() {
  __attribute__((unused)); // expected-error {{'unused' attribute cannot be applied to a statement}}
}

void reject_statement_attribute_on_declaration() {
  // expected-error@+1 {{'fallthrough' attribute cannot be applied to a declaration}}
  [[fallthrough]] int value;
}
