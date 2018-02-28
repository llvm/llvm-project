// RUN: %clang_analyze_cc1 -analyzer-checker=core %s
a;
b(void **c) { // no-crash
  *c = a;
  int *d;
  b(&d);
  *d;
}
