// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-macosx \
// RUN:   -x c-header %s -o %t/output.symbols.json -verify

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix A
void a(void) __attribute__((availability(macos, introduced=12.0)));
// A-LABEL: "!testLabel": "c:@F@a"
// A:      "availability": [
// A-NEXT:   {
// A-NEXT:     "domain": "macos",
// A-NEXT:     "introduced": {
// A-NEXT:       "major": 12,
// A-NEXT:       "minor": 0,
// A-NEXT:       "patch": 0
// A-NEXT:     }
// A-NEXT:   }
// A-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix B
void b(void) __attribute__((availability(macos, introduced=11.0, deprecated=12.0, obsoleted=20.0)));
// B-LABEL: "!testLabel": "c:@F@b"
// B:      "availability": [
// B-NEXT:   {
// B-NEXT:     "deprecated": {
// B-NEXT:       "major": 12,
// B-NEXT:       "minor": 0,
// B-NEXT:       "patch": 0
// B-NEXT:     },
// B-NEXT:     "domain": "macos",
// B-NEXT:     "introduced": {
// B-NEXT:       "major": 11,
// B-NEXT:       "minor": 0,
// B-NEXT:       "patch": 0
// B-NEXT:     },
// B-NEXT:     "obsoleted": {
// B-NEXT:       "major": 20,
// B-NEXT:       "minor": 0,
// B-NEXT:       "patch": 0
// B-NEXT:     }
// B-NEXT:   }
// B-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix E
void c(void) __attribute__((availability(macos, introduced=11.0, deprecated=12.0, obsoleted=20.0))) __attribute__((availability(ios, introduced=13.0)));
// C-LABEL: "!testLabel": "c:@F@c"
// C:       "availability": [
// C-NEXT:    {
// C-NEXT:      "deprecated": {
// C-NEXT:        "major": 12,
// C-NEXT:        "minor": 0,
// C-NEXT:        "patch": 0
// C-NEXT:      },
// C-NEXT:      "domain": "macos",
// C-NEXT:      "introduced": {
// C-NEXT:        "major": 11,
// C-NEXT:        "minor": 0,
// C-NEXT:        "patch": 0
// C-NEXT:      },
// C-NEXT:      "obsoleted": {
// C-NEXT:        "major": 20,
// C-NEXT:        "minor": 0,
// C-NEXT:        "patch": 0
// C-NEXT:      }
// C-NEXT:    }
// C-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix D
void d(void) __attribute__((deprecated)) __attribute__((availability(macos, introduced=11.0)));
// D-LABEL: "!testLabel": "c:@F@d"
// D:      "availability": [
// D-NEXT:   {
// D-NEXT:     "domain": "*",
// D-NEXT:     "isUnconditionallyDeprecated": true
// D-NEXT:   },
// D-NEXT:   {
// D-NEXT:     "domain": "macos",
// D-NEXT:     "introduced": {
// D-NEXT:       "major": 11,
// D-NEXT:       "minor": 0,
// D-NEXT:       "patch": 0
// D-NEXT:     }
// D-NEXT:   }
// D-NEXT: ]

// This symbol should be dropped as it's unconditionally unavailable
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix E
void e(void) __attribute__((unavailable)) __attribute__((availability(macos, introduced=11.0)));
// E-NOT: "!testLabel": "c:@F@e"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix F
void f(void) __attribute__((availability(macos, unavailable)));
// F-LABEL: "!testLabel": "c:@F@f"
// F:      "availability": [
// F-NEXT:   {
// F-NEXT:     "domain": "macos",
// F-NEXT:     "isUnconditionallyUnavailable": true
// F-NEXT:   }
// F-NEXT: ]

// expected-no-diagnostics

