// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin %s

// This file tests -Wconstant-conversion, a subcategory of -Wconversion
// which is on by default.

constexpr int nines() { return 99999; }

void too_big_for_char(int param) {
  char warn1 = false ? 0 : 99999;
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}
  char warn2 = false ? 0 : nines();
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}

  char warn3 = param > 0 ? 0 : 99999;
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}
  char warn4 = param > 0 ? 0 : nines();
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}

  char ok1 = true ? 0 : 99999;
  char ok2 = true ? 0 : nines();

  char ok3 = true ? 0 : 99999 + 1;
  char ok4 = true ? 0 : nines() + 1;
}
