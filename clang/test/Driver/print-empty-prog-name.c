// Test that -print-prog-name= correctly returns an empty string

// RUN: %clang -print-prog-name= 2>&1 | FileCheck %s
// CHECK-NOT:{{.+}}

