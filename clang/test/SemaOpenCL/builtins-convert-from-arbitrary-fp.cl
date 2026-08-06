// RUN: %clang_cc1 -triple spir-unknown-unknown -fsyntax-only -verify %s

void test_format_before_conversion(half *p) {
  (void)__builtin_convert_from_arbitrary_fp(*p, "Nope", float); // expected-error {{'Nope' is not a supported arbitrary floating-point format}}
}
