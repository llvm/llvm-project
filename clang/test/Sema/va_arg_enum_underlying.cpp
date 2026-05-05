// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

__builtin_va_list ap;

void foo() {
  enum E1 : char16_t {};
  enum E2 : char32_t {};
  enum E3 : unsigned char {};
  enum E4 : wchar_t {};

  (void)__builtin_va_arg(ap, E1); // expected-warning {{second argument to 'va_arg' is of promotable type}}
  (void)__builtin_va_arg(ap, E2); // expected-warning {{second argument to 'va_arg' is of promotable type}}
  (void)__builtin_va_arg(ap, E3); // expected-warning {{second argument to 'va_arg' is of promotable type}}
  (void)__builtin_va_arg(ap, E4); // expected-warning {{second argument to 'va_arg' is of promotable type}}
}
