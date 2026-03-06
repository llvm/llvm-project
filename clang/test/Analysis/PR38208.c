// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// expected-no-diagnostics

// This test case used to demonstrate a huge slowdown regression.
// Reported in https://bugs.llvm.org/show_bug.cgi?id=38208
// Caused by 2bbccca9f75b6bce08d77cf19abfb206d0c3bc2e aka. "aggressive-binary-operation-simplification"
// Fixed by dcde8acc32f1355f37d3bc2814c528fdc2ca5f94

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
