// RUN: %clang_cc1 -extract-api --pretty-sgf -triple arm64-apple-ios17.1-macabi \
// RUN:   -x c-header %s -verify -o - | FileCheck %s

int a;

// CHECK:      "platform": {
// CHECK-NEXT:   "architecture": "arm64",
// CHECK-NEXT:   "environment": "macabi",
// CHECK-NEXT:   "operatingSystem": {
// CHECK-NEXT:     "minimumVersion": {
// CHECK-NEXT:       "major": 14,
// CHECK-NEXT:       "minor": 0,
// CHECK-NEXT:       "patch": 0
// CHECK-NEXT:     },
// CHECK-NEXT:     "name": "ios"
// CHECK-NEXT:   },
// CHECK-NEXT:   "vendor": "apple"
// CHECK-NEXT: }

// expected-no-diagnostics
