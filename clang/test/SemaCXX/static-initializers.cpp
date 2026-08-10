// RUN: %clang_cc1 -fsyntax-only -verify %s 
// expected-no-diagnostics
int f() {
  return 10;
}

void g() {
  static int a = f();
}

static int b = f();
