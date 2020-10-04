// RUN: %clang_cc1 -analyze -analyzer-constraints=z3 -setup-static-analyzer \
// RUN:   -analyzer-checker=core,debug.ExprInspection %s 2>&1 | FileCheck %s
//
// REQUIRES: z3
//
// Works only with the z3 constraint manager.

void clang_analyzer_printState();

void foo(int x) {
  if (x == 3) {
    clang_analyzer_printState();
    (void)x;
    // CHECK: "constraints": [
    // CHECK-NEXT: { "symbol": "(reg_$[[#]]<int x>) == 3", "range": "(= reg_$[[#]] #x00000003)" }
  }
}
