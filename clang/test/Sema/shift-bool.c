// RUN: %clang_cc1 -fsyntax-only -Wshift-bool -verify %s

void t() {
  int x = 10;
  int y = 1;

  int a = y << x;
  int b = y >> x;

  int c = 0 << x;
  int d = 0 >> x;

  int e = y << 1;
  int f = y >> 1;

  int g = y << -1; // expected-warning {{shift count is negative}}
  int h = y >> -1; // expected-warning {{shift count is negative}}

  int i = y << 0;
  int j = y >> 0;

  if ((y << 1) != 0) { }
  if ((y >> 1) != 0) { }
}
