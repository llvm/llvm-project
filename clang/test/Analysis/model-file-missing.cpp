// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-config model-path=%t/blah %s -o - 2>&1 | FileCheck %s
// CHECK: error: invalid input for analyzer-config option 'model-path', that expects a filename value
// CHECK-NEXT: 1 error generated
