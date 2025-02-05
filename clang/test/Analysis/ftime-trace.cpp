// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -ftime-trace=%t.raw.json -ftime-trace-granularity=0 -verify
// RUN: %python -c 'import json, sys; print(json.dumps(json.load(sys.stdin), indent=4))' < %t.raw.json > %t.formatted.json
// RUN: FileCheck --input-file=%t.formatted.json --check-prefix=CHECK %s

// The trace file is rather large, but it should contain at least the duration of the analysis of 'f':
//
// CHECK:          "name": "HandleCode f",
// CHECK-NEXT:     "args": {
// CHECK-NEXT:         "detail": "f()",
// CHECK-NEXT:         "file": "{{.+}}ftime-trace.cpp",
// CHECK-NEXT:         "line": {{[0-9]+}}
// CHECK-NEXT:     }

// If any reports are found, "flushing" their equivalence class (EQC) is a separate action:
//
// CHECK:          "name": "Flushing EQC Division by zero",
// CHECK-NEXT:     "args": {
// CHECK-NEXT:         "detail": "core.DivideZero",
// CHECK-NEXT:         "file": "{{.+}}ftime-trace.cpp",
// CHECK-NEXT:         "line": {{[0-9]+}}
// CHECK-NEXT:     }

// The trace also contains durations of each step, but they are so short that they are not reliably present
// in each run. However, they are also aggregated into Total *, for example:
//
// CHECK:          "name": "Total Loc PostStmt",
// CHECK-NEXT:     "args": {
// CHECK-NEXT:         "count": {{[0-9]+}},
// CHECK-NEXT:         "avg ms": {{[0-9]+}}
// CHECK-NEXT:     }

// Additionally, the trace lists checker hook points (again, relying on totals here):
//
// CHECK:          "name": "Total CheckerManager::runCheckersForStmt (Pre)",
// CHECK-NEXT:     "args": {
// CHECK-NEXT:         "count": {{[0-9]+}},
// CHECK-NEXT:         "avg ms": {{[0-9]+}}
// CHECK-NEXT:     }

// Finally, each checker call back is also present:
//
// CHECK:          "name": "Total Stmt:core.DivideZero",
// CHECK-NEXT:     "args": {
// CHECK-NEXT:         "count": {{[0-9]+}},
// CHECK-NEXT:         "avg ms": {{[0-9]+}}
// CHECK-NEXT:     }

bool coin();
int f() {
    int x = 0;
    int y = 0;
    while (coin()) {
        x = 1;
    }
    return x / y; // expected-warning{{Division by zero}}
}
