// RUN: %clang_analyze_cc1 -analyzer-checker=core %s -ftime-trace=%t.raw.json -ftime-trace-granularity=0 -verify
// RUN: %python -c 'import json, sys; print(json.dumps(json.load(sys.stdin), indent=4))' < %t.raw.json > %t.formatted.json
// RUN: FileCheck --input-file=%t.formatted.json --check-prefix=CHECK %s

// The trace file is rather large, but it should contain at least one scope for removeDead:
//
// CHECK:          "name": "ExprEngine::removeDead"

bool coin();
int f() {
    int x = 0;
    int y = 0;
    while (coin()) {
        x = 1;
    }
    return x / y; // expected-warning{{Division by zero}}
}
