// RUN: %clang_cc1 -triple mips64 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mips64 -target-feature +soft-float -fsyntax-only -verify=softfloat %s

// expected-no-diagnostics

void test_f(float p) {
  float result = p;
  __asm__("" :: "f"(result)); // softfloat-error{{invalid input constraint 'f' in asm}}
}
