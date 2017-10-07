// RUN: %clang_cc1 -fsyntax-only -fwchar-type=short -fno-signed-wchar -verify %s

void f() {
  (void)L"\U00010000"; // unicode escape produces UTF-16 sequence, so no warning

  (void)L'\U00010000'; // expected-error {{character too large for enclosing character literal type}}

  (void)L'ab';  // expected-warning {{extraneous characters in character constant ignored}}

  (void)L'a\u1000';  // expected-warning {{extraneous characters in character constant ignored}}
}

