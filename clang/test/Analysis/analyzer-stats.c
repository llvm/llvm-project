// RUN: %clang_analyze_cc1 -analyzer-checker=core,deadcode.DeadStores,debug.Stats -verify -Wno-unreachable-code -analyzer-opt-analyze-nested-blocks %s

int foo();

int test() { // expected-warning-re{{test -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 0 | Exhausted Block: no | Empty WorkList: yes}}
  int a = 1;
  a = 34 / 12;

  if (foo())
    return a;

  a /= 4;
  return a;
}
