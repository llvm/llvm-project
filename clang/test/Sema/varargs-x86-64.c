// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-apple-darwin9

// rdar://6726818
void f1() {
  const __builtin_va_list args2;
  (void)__builtin_va_arg(args2, int); // expected-error {{first argument to 'va_arg' is of type 'const __builtin_va_list' and not 'va_list'}}
}

