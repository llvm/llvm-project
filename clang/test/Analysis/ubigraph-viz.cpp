// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.API -analyzer-viz-egraph-ubigraph -verify %s
// expected-no-diagnostics

int f(int x) {
  return x < 0 ? 0 : 42;
}

