// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -DUNSIGNED -Wsign-conversion %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -verify -Wsign-conversion %s

// PR35200
enum X { A,B,C};
int f(enum X x) {
#ifdef UNSIGNED
  return x; // expected-warning {{implicit conversion changes signedness: 'enum X' to 'int'}}
#else
  // expected-no-diagnostics
  return x;
#endif
}
