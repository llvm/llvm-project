// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,deadcode.DeadStores,debug.Stats \
// RUN:   -Wno-unreachable-code \
// RUN:   -verify=default %s

// NOTE: analyzer-max-loop option is only meaningful if unroll-loops is false,
//       that's why we do not pass it in the first case, as unroll-loops is
//       true by default.

// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-config unroll-loops=false \
// RUN:   -analyzer-max-loop 4 \
// RUN:   -analyzer-checker=core,deadcode.DeadStores,debug.Stats \
// RUN:   -Wno-unreachable-code \
// RUN:   -verify=no-unroll %s

int foo(void);

int test(void) { // default-warning-re{{test -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 0 | Exhausted Block: no | Empty WorkList: yes}}
                 // no-unroll-warning-re@-1{{test -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 0 | Exhausted Block: no | Empty WorkList: yes}}
  int a = 1;
  a = 34 / 12;

  if (foo())
    return a;

  a /= 4;
  return a;
}


int sink(void) // default-warning-re{{sink -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 0 | Exhausted Block: no | Empty WorkList: yes}}
{              // no-unroll-warning-re@-1{{sink -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 1 | Exhausted Block: yes | Empty WorkList: yes}}
  for (int i = 0; i < 10; ++i) // no-unroll-warning {{(sink): The analyzer generated a sink at this point}}
    ++i;

  return 0;
}

int emptyConditionLoop(void) // default-warning-re{{emptyConditionLoop -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 1 | Exhausted Block: yes | Empty WorkList: yes}}
{                            // no-unroll-warning-re@-1{{emptyConditionLoop -> Total CFGBlocks: {{[0-9]+}} | Unreachable CFGBlocks: 1 | Exhausted Block: yes | Empty WorkList: yes}}
  int num = 1;
  for (;;)  // Infinite loop - cannot be unrolled (no compile-time bound)
            // Both with and without unrolling: 1 unreachable block (the exit after loop)
    num++;
}
