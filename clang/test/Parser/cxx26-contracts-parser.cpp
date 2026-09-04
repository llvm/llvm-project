// RUN: %clang_cc1 -std=c++2c -fcontracts -fsyntax-only -verify %s

// This test intentionally exercises parser-only support. Contract predicates
// are parsed as ordinary expressions, but are not retained in the AST yet.

int pre = 42;
int post(int value) { return value; }

int divide(int a, int b) pre(b != 0);
int square(int x) post(result: result >= x);
int clamp(int x) pre(x >= 0) pre(x <= 100)
    post(result: x >= 0) post(other_result: x <= 100);

struct OuterResult {};
OuterResult shadowed_result;
int result_shadowing(int x)
    post(shadowed_result: shadowed_result >= x);

template <typename T>
concept C = true;

template <typename T>
int constrained(T value) requires C<T> pre(value > T{});

struct S {
  int member(int value) pre(value > 0) post(result: result > value);
  virtual int virtual_member(int value) const final pre(value > 0);
};

int S::member(int value) pre(value > 0) post(result: result > value) {
  contract_assert(value > 0);
  contract_assert [[maybe_unused]] (value < 100);
  return value;
}

void lambdas() {
  auto a = [](int value) pre(value > 0) { return value; };
  auto b = [] pre(true) { return 0; };
  (void)a;
  (void)b;
}

int attributed(int value)
    pre [[maybe_unused]] (value > 0)
    post [[maybe_unused]] (result: value > 0);

int not_a_function pre(true); // expected-error {{contract specifiers can only be applied to function declarations}}

template <typename T>
int wrong_order(T value) pre(value > T{}) requires C<T>;
// expected-error@-1 {{trailing requires clause must appear before contract specifiers}}

void missing_assert_lparen() {
  contract_assert true; // expected-error {{expected '(' after 'contract_assert'}}
}

int missing_pre_lparen() pre true;
// expected-error@-1 {{expected '(' after 'pre'}}
// expected-error@-2 {{expected function body after function declarator}}
