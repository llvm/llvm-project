// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-linux-gnu -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-linux-gnu -verify %s

extern const char *str;

int main(void) {
  int r = 0;

  r |= __builtin_allow_runtime_check(); // expected-error {{too few arguments to function call}}

  r |= __builtin_allow_runtime_check(str); // expected-error {{expression is not a string literal}}

  r |= __builtin_allow_runtime_check(5); // expected-error {{incompatible integer to pointer conversion}} expected-error {{expression is not a string literal}}

  r |= __builtin_allow_runtime_check("a", "b");  // expected-error {{too many arguments to function call}}

  r |= __builtin_allow_runtime_check("");

  r |= __builtin_allow_runtime_check("check");

  str = __builtin_allow_runtime_check("check2");  // expected-error {{incompatible integer to pointer conversion}}

  return r;
}
