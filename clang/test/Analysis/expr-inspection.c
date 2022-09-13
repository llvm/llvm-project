// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=debug.ExprInspection \
// RUN:  -verify %s 2>&1 | FileCheck %s

// Self-tests for the debug.ExprInspection checker.

void clang_analyzer_dump(int x);
void clang_analyzer_dump_pointer(int *p);
void clang_analyzer_dumpSvalType(int x);
void clang_analyzer_dumpSvalType_pointer(int *p);
void clang_analyzer_printState(void);
void clang_analyzer_numTimesReached(void);

void foo(int x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump(x + (-1)); // expected-warning{{(reg_$0<int x>) - 1}}
  clang_analyzer_dumpSvalType(x); // expected-warning {{int}}

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
// CHECK-NEXT:   "store": { "pointer": "{{0x[0-9a-f]+}}", "items": [
// CHECK-NEXT:     { "cluster": "y", "pointer": "{{0x[0-9a-f]+}}", "items": [
// CHECK-NEXT:       { "kind": "Direct", "offset": {{[0-9]+}}, "value": "2 S32b" }
// CHECK-NEXT:     ]}
// CHECK-NEXT:   ]},
// CHECK-NEXT:   "environment": { "pointer": "{{0x[0-9a-f]+}}", "items": [
// CHECK-NEXT:     { "lctx_id": {{[0-9]+}}, "location_context": "#0 Call", "calling": "foo", "location": null, "items": [
// CHECK-NEXT:       { "stmt_id": {{[0-9]+}}, "kind": "ImplicitCastExpr", "pretty": "clang_analyzer_printState", "value": "&code{clang_analyzer_printState}" }
// CHECK-NEXT:     ]}
// CHECK-NEXT:   ]},
// CHECK-NEXT:   "constraints": [
// CHECK-NEXT:     { "symbol": "reg_$0<int x>", "range": "{ [-2147483648, 13] }" }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "equivalence_classes": null,
// CHECK-NEXT:   "disequality_info": null,
// CHECK-NEXT:   "dynamic_types": null,
// CHECK-NEXT:   "dynamic_casts": null,
// CHECK-NEXT:   "checker_messages": null
// CHECK-NEXT: }

struct S {
  int x, y;
};

void test_field_dumps(struct S s, struct S *p) {
  clang_analyzer_dump_pointer(&s.x); // expected-warning{{&s.x}}
  clang_analyzer_dump_pointer(&p->x); // expected-warning{{&Element{SymRegion{reg_$1<struct S * p>},0 S64b,struct S}.x}}
  clang_analyzer_dumpSvalType_pointer(&s.x); // expected-warning {{int *}}
}
