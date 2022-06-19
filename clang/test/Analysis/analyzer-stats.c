// RUN: %clang_analyze_cc1 -analyzer-checker=core,deadcode.DeadStores,debug.Stats -verify -Wno-unreachable-code -analyzer-max-loop 4 %s

int foo(void);

int test(void) { // expected-warning-re{{test -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 0 | Exhausted Block: no | Empty WorkList: yes}}
  int a = 1;
  a = 34 / 12;

  if (foo())
    return a;

  a /= 4;
  return a;
}


int sink(void) // expected-warning-re{{sink -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 1 | Exhausted Block: yes | Empty WorkList: yes}}
{
  for (int i = 0; i < 10; ++i) // expected-warning {{(sink): The analyzer generated a sink at this point}}
    ++i;

  return 0;
}

int emptyConditionLoop(void) // expected-warning-re{{emptyConditionLoop -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 0 | Exhausted Block: yes | Empty WorkList: yes}}
{
  int num = 1;
  for (;;)
    num++;
}
