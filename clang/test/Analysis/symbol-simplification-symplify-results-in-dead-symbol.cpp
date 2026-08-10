// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

#define assert(cond) if (!(cond)) return

unsigned a, b;
void f(unsigned c) {
  assert(c == b);
  assert((c | a) != a);
  assert(a); // no-crash
}
