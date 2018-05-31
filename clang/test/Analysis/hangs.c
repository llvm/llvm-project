// RUN: %clang_analyze_cc1 -analyzer-checker core -verify %s

// expected-no-diagnostics

// Stuff that used to hang.

int g();

int f(int y) {
  return y + g();
}

int produce_a_very_large_symbol(int x) {
  return f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(
             f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(x))))))))))))))))))))))))))))))));
}

void produce_an_exponentially_exploding_symbol(int x, int y) {
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
}
