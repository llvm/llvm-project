// RUN: %clang_cc1 -analyze -analyzer-checker=core,deadcode.DeadStores,debug.Stats -verify -Wno-unreachable-code -analyzer-opt-analyze-nested-blocks %s

int foo();

int test() { // expected-warning{{Total CFGBlocks}}
  int a = 1;
  a = 34 / 12;

  if (foo())
    return a;

  a /= 4;
  return a;
}
