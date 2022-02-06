// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Check that we don't get any extra warning for "return" without an
// expression, in a function that might have been intended to return
// void all along.
auto f1() {
  return 1;
  return; // expected-error {{deduced as 'void' here but deduced as 'int' in earlier return statement}}
}

decltype(auto) f2() {
  return 1;
  return; // expected-error {{deduced as 'void' here but deduced as 'int' in earlier return statement}}
}

auto *g() {
  return; // expected-error {{cannot deduce return type 'auto *' from omitted return expression}}
}

decltype(h1) h1() { // expected-error {{use of undeclared identifier 'h1'}}
  return;
}
