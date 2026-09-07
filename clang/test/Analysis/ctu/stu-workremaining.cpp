// RUN: %clang_analyze_cc1 -std=c++20 \
// RUN:   -analyzer-checker=core,alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config max-nodes=10 \
// RUN:   -verify=ctu-on %s

// RUN: %clang_analyze_cc1 -std=c++20 \
// RUN:   -analyzer-checker=core,alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=false \
// RUN:   -analyzer-config max-nodes=10 \
// RUN:   -verify=ctu-off %s

#define NOP ((void)0)

void tp(int x) {
  // ctu-on-warning@+4{{This statement is never executed}}
  // ctu-on-warning@+3{{self-comparison always evaluates to false}}
  // ctu-off-warning@+2{{This statement is never executed}}
  // ctu-off-warning@+1{{self-comparison always evaluates to false}}
  if (x != x) NOP;
}

void fp(int x) {
  NOP; NOP; NOP; NOP; NOP;
  NOP; NOP; NOP; NOP; NOP;
  if (x) NOP; // no-warning: the true branch might be alive even in CTU
}
