// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   2>&1 | FileCheck %s

// In this test we check whether the solver's symbol simplification mechanism
// is capable of reaching a fixpoint. This should be done after one iteration.

void clang_analyzer_printState();

void test(int a, int b, int c) {
  if (a + b != c)
    return;
  clang_analyzer_printState();
  // CHECK:      "constraints": [
  // CHECK-NEXT:   { "symbol": "((reg_$0<int a>) + (reg_$1<int b>)) != (reg_$2<int c>)", "range": "{ [0, 0] }" }
  // CHECK-NEXT: ],
  // CHECK-NEXT: "equivalence_classes": [
  // CHECK-NEXT:   [ "(reg_$0<int a>) + (reg_$1<int b>)", "reg_$2<int c>" ]
  // CHECK-NEXT: ],
  // CHECK-NEXT: "disequality_info": null,

  // Simplification starts here.
  if (b != 0)
    return;
  clang_analyzer_printState();
  // CHECK:        "constraints": [
  // CHECK-NEXT:     { "symbol": "(reg_$0<int a>) != (reg_$2<int c>)", "range": "{ [0, 0] }" },
  // CHECK-NEXT:     { "symbol": "reg_$1<int b>", "range": "{ [0, 0] }" }
  // CHECK-NEXT:   ],
  // CHECK-NEXT:   "equivalence_classes": [
  // CHECK-NEXT:     [ "(reg_$0<int a>) != (reg_$2<int c>)" ],
  // CHECK-NEXT:     [ "reg_$0<int a>", "reg_$2<int c>" ]
  // CHECK-NEXT:   ],
  // CHECK-NEXT: "disequality_info": null,

  // Keep the symbols and the constraints! alive.
  (void)(a * b * c);
  return;
}
