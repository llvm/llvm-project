// RUN: %clang_cc1 -std=c++2c -fcontracts -fsyntax-only -verify %s

template <typename>
concept Recoverable = true;

int empty_pre(int value) pre(); // expected-error {{expected expression}}
int after_empty_pre(int value) pre(value > 0);

int empty_post(int value) post(result:); // expected-error {{expected expression}}
int after_empty_post(int value) post(value > 0);

int missing_pre_close(int value) pre(value > 0; // expected-error {{expected ')'}} expected-note {{to match this '('}}
int after_missing_pre_close(int value) pre(value > 0);

void assertion_recovery(int value) {
  contract_assert(); // expected-error {{expected expression}}
  int after_empty = value;

  contract_assert value > 0; // expected-error {{expected '(' after 'contract_assert'}}
  int after_missing_open = after_empty;

  contract_assert((value > 0) && ((value + 1) > 1));

  contract_assert(value > 0; // expected-error {{expected ')'}} expected-note {{to match this '('}}
  int after_missing_close = value;
  (void)after_missing_close;
  (void)after_missing_open;
}

template <typename T>
int wrong_order_declaration(T value) pre(value > T{}) requires Recoverable<T>;
// expected-error@-1 {{trailing requires clause must appear before contract specifiers}}

template <typename T>
int wrong_order_definition(T value) pre(value > T{}) requires Recoverable<T> {
  // expected-error@-1 {{trailing requires clause must appear before contract specifiers}}
  return value;
}

struct RecoveryMember {
  template <typename T>
  int wrong_order(T value) pre(value > T{}) requires Recoverable<T>;
  // expected-error@-1 {{trailing requires clause must appear before contract specifiers}}

  int after_error(int value) pre(value > 0);
};

int missing_pre_lparen(int value) pre value > 0;
// expected-error@-1 {{expected '(' after 'pre'}}
// expected-error@-2 {{expected function body after function declarator}}

int after_missing_pre_lparen(int value) post(value > 0);
