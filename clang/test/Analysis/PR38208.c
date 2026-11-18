// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config aggressive-binary-operation-simplification=true -verify %s

// expected-no-diagnostics

// This test case used to demonstrate a huge slowdown regression.
// Reported in https://bugs.llvm.org/show_bug.cgi?id=38208
// Caused by bad logic in 2bbccca9f75b6bce08d77cf19abfb206d0c3bc2e aka.
// "aggressive-binary-operation-simplification", which created symbols without
// respecting the symbol complexity limit. This was originally avoided by
// disabling that feature, but now the underlying cause is also eliminated.

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
