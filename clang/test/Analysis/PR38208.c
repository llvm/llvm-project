// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// expected-no-diagnostics

int foo(int a, int b) {
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  a += b; b -= a;
  return a + b;
}
