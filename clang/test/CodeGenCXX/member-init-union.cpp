// RUN: %clang_cc1 %s -emit-llvm-only -verify
// expected-no-diagnostics

union x {
  int a;
  float b;
  x(float y) : b(y) {}
  x(int y) : a(y) {}
};
x a(1), b(1.0f);

