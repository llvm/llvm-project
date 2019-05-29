// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=debug.ExprInspection \
// RUN:  -verify %s 2>&1 | FileCheck %s

// Self-tests for the debug.ExprInspection checker.

void clang_analyzer_dump(int x);
void clang_analyzer_printState();
void clang_analyzer_numTimesReached();

void foo(int x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump(x + (-1)); // expected-warning{{(reg_$0<int x>) + -1}}
  int y = 1;
  for (; y < 3; ++y) {
    clang_analyzer_numTimesReached(); // expected-warning{{2}}

    if (y == 2) {
      int z = x > 13;
      if (!z)
        clang_analyzer_printState();
    }
  }
}

// CHECK:      "program_state": {
// CHECK-NEXT:   "store": [
// CHECK-NEXT:     { "cluster": "y", "items": [
// CHECK-NEXT:       { "kind": "Direct", "offset": 0, "value": "2 S32b" }
// CHECK-NEXT:     ]}
// CHECK-NEXT:   ],
// CHECK-NEXT:   "environment": [
// CHECK-NEXT:     { "location_context": "#0 Call", "calling": "foo", "call_line": null, "items": [
// CHECK-NEXT:       { "lctx_id": 1, "stmt_id": {{[0-9]+}}, "pretty": "clang_analyzer_printState", "value": "&code{clang_analyzer_printState}" }
// CHECK-NEXT:     ]}
// CHECK-NEXT:   ],
// CHECK-NEXT:   "constraints": [
// CHECK-NEXT:     { "symbol": "reg_$0<int x>", "range": "{ [-2147483648, 13] }" }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "dynamic_types": null,
// CHECK-NEXT:   "constructing_objects": null,
// CHECK-NEXT:   "checker_messages": null
// CHECK-NEXT: }

