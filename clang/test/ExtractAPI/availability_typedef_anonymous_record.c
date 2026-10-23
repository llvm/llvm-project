// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-macosx \
// RUN:   -x c-header %s -o %t/output.symbols.json -verify

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix STRUCT

typedef struct {
  int x;
} MyStruct __attribute__((availability(macos, introduced=12.0)));

// STRUCT-LABEL: "!testLabel": "c:@SA@MyStruct"
// STRUCT:      "availability": [
// STRUCT-NEXT:   {
// STRUCT-NEXT:     "domain": "macos",
// STRUCT-NEXT:     "introduced": {
// STRUCT-NEXT:       "major": 12,
// STRUCT-NEXT:       "minor": 0,
// STRUCT-NEXT:       "patch": 0
// STRUCT-NEXT:     }
// STRUCT-NEXT:   }
// STRUCT-NEXT: ]
// STRUCT:      "title": "MyStruct"

// expected-no-diagnostics
