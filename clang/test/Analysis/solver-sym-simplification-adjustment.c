// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

void clang_analyzer_warnIfReached();
void clang_analyzer_eval();

void test_simplification_adjustment_concrete_int(int b, int c) {
  if (b < 0 || b > 1)  // b: [0,1]
    return;
  if (c < -1 || c > 1) // c: [-1,1]
    return;
  if (c + b != 0)      // c + b == 0
    return;
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  if (b != 1)          // b == 1  --> c + 1 == 0 --> c == -1
    return;
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  clang_analyzer_eval(c == -1);   // expected-warning{{TRUE}}

  // Keep the symbols and the constraints! alive.
  (void)(b * c);
  return;
}

void test_simplification_adjustment_range(int b, int c) {
  if (b < 0 || b > 1)              // b: [0,1]
    return;
  if (c < -1 || c > 1)             // c: [-1,1]
    return;
  if (c + b < -1 || c + b > 0)     // c + b: [-1,0]
    return;
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  if (b != 1)                      // b == 1  --> c + 1: [-1,0] --> c: [-2,-1]
    return;
                                   // c: [-2,-1] is intersected with the
                                   // already associated range which is [-1,1],
                                   // thus we get c: [-1,-1]
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  clang_analyzer_eval(c == -1);    // expected-warning{{TRUE}}

  // Keep the symbols and the constraints! alive.
  (void)(b * c);
  return;
}

void test_simplification_adjustment_to_infeasible_concrete_int(int b, int c) {
  if (b < 0 || b > 1) // b: [0,1]
    return;
  if (c < 0 || c > 1) // c: [0,1]
    return;
  if (c + b != 0)     // c + b == 0
    return;
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  if (b != 1) {       // b == 1  --> c + 1 == 0 --> c == -1 contradiction
    clang_analyzer_eval(b == 0);  // expected-warning{{TRUE}}
    clang_analyzer_eval(c == 0);  // expected-warning{{TRUE}}
    // Keep the symbols and the constraints! alive.
    (void)(b * c);
    return;
  }
  clang_analyzer_warnIfReached(); // no warning

  // Keep the symbols and the constraints! alive.
  (void)(b * c);
  return;
}

void test_simplification_adjustment_to_infeassible_range(int b, int c) {
  if (b < 0 || b > 1)              // b: [0,1]
    return;
  if (c < 0 || c > 1)              // c: [0,1]
    return;
  if (c + b < -1 || c + b > 0)     // c + b: [-1,0]
    return;
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  if (b != 1)                      // b == 1  --> c + 1: [-1,0] --> c: [-2,-1] contradiction
    return;
  clang_analyzer_warnIfReached();  // no warning

  // Keep the symbols and the constraints! alive.
  (void)(b * c);
  return;
}

void test_simplification_adjusment_no_infinite_loop(int a, int b, int c) {
  if (a == b)        // a != b
    return;
  if (c != 0)        // c == 0
    return;

  if (b != 0)        // b == 0
    return;
  // The above simplification of `b == 0` could result in an infinite loop
  // unless we detect that the State is unchanged.
  // The loop:
  // 1) Simplification of the trivial equivalence class
  //      "symbol": "(reg_$0<int a>) == (reg_$1<int b>)", "range": "{ [0, 0] }"
  //    results in
  //      "symbol": "(reg_$0<int a>) == 0", "range": "{ [0, 0] }" }
  //    which in turn creates a non-trivial equivalence class
  //      [ "(reg_$0<int a>) == (reg_$1<int b>)", "(reg_$0<int a>) == 0" ]
  // 2) We call assumeSymInclusiveRange("(reg_$0<int a>) == 0")
  //    and that calls **simplify** on the associated non-trivial equivalence
  //    class. During the simplification the State does not change, we reached
  //    the fixpoint.

  (void)(a * b * c);
}
