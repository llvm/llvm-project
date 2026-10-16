// RUN: %clang_cc1 -fsyntax-only -Wshift-bool -verify %s

void t() {
  int x = 10;
  int y = 5;

  int a = (x < y) << 1;
  int b = (x < y) >> 1;

  int c = (x > y) << 1;
  int d = (x > y) >> 1;

  int e = (x == y) << 1;
  int f = (x == y) >> 1;

  int g = (x != y) << 1;
  int h = (x != y) >> 1;

  int i = (x < y) << 0;
  int j = (x < y) >> 0;

  int k = (x < y) << -1; // expected-warning {{shift count is negative}}
  int l = (x < y) >> -1; // expected-warning {{shift count is negative}}

  if (((x < y) << 1) != 0) { }
  if (((x < y) >> 1) != 0) { }
}
