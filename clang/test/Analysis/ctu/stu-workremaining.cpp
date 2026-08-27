// RUN: %clang_analyze_cc1 -std=c++20 \
// RUN:   -analyzer-checker=core,alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config max-nodes=10 \
// RUN:   -verify=ctu-on %s
// ctu-on-no-diagnostics

// RUN: %clang_analyze_cc1 -std=c++20 \
// RUN:   -analyzer-checker=core,alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=false \
// RUN:   -analyzer-config max-nodes=10 \
// RUN:   -verify=ctu-off %s
// ctu-off-no-diagnostics

#define NOP ((void)0)

void entrypoint(int x) {
  NOP; NOP; NOP; NOP; NOP;
  NOP; NOP; NOP; NOP; NOP;
  if (x) NOP;
}
