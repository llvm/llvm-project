// RUN: %clang_cc1 -verify -std=c++11 -fsyntax-only %s

[[noreturn]] void throw_int() {
  throw int(); // expected-error {{cannot use 'throw' with exceptions disabled}}
}

void throw_int_wrapper() {
  [[clang::musttail]] return throw_int(); // expected-error {{'musttail' attribute may not be used with no-return-attribute functions}}
}
