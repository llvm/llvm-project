// RUN: %clang_cc1 -verify -std=c++2a -fsyntax-only %s

auto f() = delete; // expected-note 2 {{candidate function has been explicitly deleted}}

auto g1() {
  auto x = f(); // expected-error {{call to deleted function 'f'}}
  return x;
}

auto g2() {
  decltype(auto) x = f(); // expected-error {{call to deleted function 'f'}}
  return x;
}
