// RUN: %clang_cc1 -fsyntax-only -verify %s

auto f() = delete; // expected-note {{candidate function has been explicitly deleted}}

auto g() {
  auto x = f(); // expected-error {{call to deleted function 'f'}}
  return x;
}
