// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s

// va_list on ARM AAPCS is struct { void* __ap }.
int test1(const __builtin_va_list &ap) {
  return __builtin_va_arg(ap, int); // expected-error {{binding value of type 'const __builtin_va_list' to reference to type '__builtin_va_list' drops 'const' qualifier}}
}
