// RUN: %clang_cc1 -fsyntax-only -fwchar-type=short -fno-signed-wchar -verify %s

void f(void) {
  (void)L"\U00010000"; // unicode escape produces UTF-16 sequence, so no warning

  (void)L'ab';  // expected-error {{wide character literals may not contain multiple characters}}

  (void)L'a\u1000';  // expected-error {{wide character literals may not contain multiple characters}}
}

