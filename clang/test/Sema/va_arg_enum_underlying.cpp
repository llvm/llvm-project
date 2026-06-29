// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

__builtin_va_list ap;

void foo() {
  __builtin_va_arg(ap, char8_t);  // expected-warning {{second argument to 'va_arg' is of promotable type}}
  __builtin_va_arg(ap, char16_t); // expected-warning {{second argument to 'va_arg' is of promotable type}}
  __builtin_va_arg(ap, char32_t); // expected-warning {{second argument to 'va_arg' is of promotable type}}
}
