// RUN: %clang_cc1 -Wunsafe-buffer-usage -verify %s

void testIncrement(char *p) {
  ++p; // expected-warning{{unchecked operation on raw buffer in expression}}
  p++; // expected-warning{{unchecked operation on raw buffer in expression}}
  --p; // expected-warning{{unchecked operation on raw buffer in expression}}
  p--; // expected-warning{{unchecked operation on raw buffer in expression}}
}
