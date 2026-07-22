// RUN: %clang_cc1 -fsyntax-only -verify %s

auto f() = delete; // expected-note 2 {{candidate function has been explicitly deleted}}

auto g() {
  auto x = f(); // expected-error {{call to deleted function 'f'}}
  return x;
}

auto g2() {
  decltype(auto) x = f(); // expected-error {{call to deleted function 'f'}}
  return x;
}
