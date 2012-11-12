// RUN: %clang_cc1 -isystem %S/Inputs -verify %s
#include <array-bounds-system-header.h>
void test_system_header_macro() {
  BAD_MACRO_1; // no-warning
  char a[3]; // expected-note 2 {{declared here}}
  BAD_MACRO_2(a, 3); // expected-warning {{array index 3}}
  QUESTIONABLE_MACRO(a);
  NOP(a[3] = 5); // expected-warning {{array index 3}}
}
