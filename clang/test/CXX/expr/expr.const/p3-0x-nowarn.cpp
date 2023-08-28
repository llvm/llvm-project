// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wno-c++11-narrowing -verify %s
// expected-no-diagnostics

void f(int x) {
  switch (x) {
    case 0x80000001: break;
  }
}
