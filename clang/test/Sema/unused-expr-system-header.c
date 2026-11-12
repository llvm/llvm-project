// RUN: %clang_cc1 -isystem %S/Inputs -fsyntax-only -verify %s
#include <unused-expr-system-header.h>
void f(int i1, int i2) {
  POSSIBLY_BAD_MACRO(5);
  STATEMENT_EXPR_MACRO(5);
  COMMA_MACRO_1(i1 == i2, f(i1, i2)); // expected-warning {{comparison result unused}} \
                                      // expected-note {{equality comparison}}
  COMMA_MACRO_2(i1 == i2, f(i1, i2));
  COMMA_MACRO_3(i1 == i2, f(i1, i2)); // expected-warning {{comparison result unused}} \
                                      // expected-note {{equality comparison}}
  COMMA_MACRO_4(i1 == i2, f(i1, i2));
}
