// RUN: %clang_cc1 -std=c++2c -fcontracts -fsyntax-only -verify %s

int divide(int a, int b) pre(b != 0);

int square(int x) post(r: r >= 0);

int abs_val(int x) pre(x >= -1000) pre(x <= 1000) post(r: r >= 0);

int safe_div(int a, int b) pre(b != 0) post(r: r * b == a);

// Multiple post with result names
int clamp(int x) post(r: r >= 0) post(r: r <= 100);

// post without result name
int identity(int x) post(x >= 0);

void f(int x) {
  contract_assert(x > 0);
}

// pre and post are not keywords - they can still be used as identifiers.
int pre = 42;
int post(int x) { return x; }
void use() {
  pre = 10;
  post(5);
}

// post(name: expr) on void-returning function is an error.
void g() post(r: r > 0); // expected-error {{post-condition result name on function returning void}} \
                          // expected-error {{use of undeclared identifier 'r'}}

