// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -x c++ %s

// This file is encoded using ISO-8859-1

int main() {
  (void)'י'; // expected-warning {{illegal character encoding in character literal}}
  (void)u'י'; // expected-error {{illegal character encoding in character literal}}
  (void)U'י'; // expected-error {{illegal character encoding in character literal}}
  (void)L'י'; // expected-error {{illegal character encoding in character literal}}

  // For narrow character literals, since there is no error, make sure the
  // encoding is correct
  static_assert((unsigned char)'י' == 0xE9, ""); // expected-warning {{illegal character encoding in character literal}}
  static_assert('יי' == 0xE9E9, ""); // expected-warning {{illegal character encoding in character literal}} expected-warning {{multi-character character constant}}
}
