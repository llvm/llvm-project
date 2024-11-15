// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   2>&1 | FileCheck %s

// In this test we check whether the solver's symbol simplification mechanism
// is capable of reaching a fixpoint. This should be done after TWO iterations.

void clang_analyzer_printState();

void test(int a, int b, int c, int d) {
  if (a + b + c != d)
    return;
  if (c + b != 0)
    return;
  clang_analyzer_printState();
  // CHECK:      "constraints": [
  // CHECK-NEXT:   { "symbol": "(((reg_$0<int a>) + (reg_$1<int b>)) + (reg_$2<int c>)) != (reg_$3<int d>)", "range": "{ [0, 0] }" },
  // CHECK-NEXT:   { "symbol": "(reg_$2<int c>) + (reg_$1<int b>)", "range": "{ [0, 0] }" }
  // CHECK-NEXT: ],
  // CHECK-NEXT: "equivalence_classes": [
  // CHECK-NEXT:   [ "((reg_$0<int a>) + (reg_$1<int b>)) + (reg_$2<int c>)", "reg_$3<int d>" ]
  // CHECK-NEXT: ],
  // CHECK-NEXT: "disequality_info": null,

  // Simplification starts here.
  if (b != 0)
    return;
  clang_analyzer_printState();
  // CHECK:       "constraints": [
  // CHECK-NEXT:    { "symbol": "(reg_$0<int a>) != (reg_$3<int d>)", "range": "{ [0, 0] }" },
  // CHECK-NEXT:    { "symbol": "reg_$1<int b>", "range": "{ [0, 0] }" },
  // CHECK-NEXT:    { "symbol": "reg_$2<int c>", "range": "{ [0, 0] }" }
  // CHECK-NEXT:  ],
  // CHECK-NEXT:  "equivalence_classes": [
  // CHECK-NEXT:    [ "(reg_$0<int a>) != (reg_$3<int d>)" ],
  // CHECK-NEXT:    [ "reg_$0<int a>", "reg_$3<int d>" ],
  // CHECK-NEXT:    [ "reg_$2<int c>" ]
  // CHECK-NEXT:  ],
  // CHECK-NEXT:  "disequality_info": null,

  // Keep the symbols and the constraints! alive.
  (void)(a * b * c * d);
  return;
}
