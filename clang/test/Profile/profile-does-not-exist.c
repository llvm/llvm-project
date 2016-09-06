// RUN: not %clang_cc1 -emit-llvm %s -o - -fprofile-instrument-use-path=%t.nonexistent.profdata 2>&1 | FileCheck %s

// CHECK: error: Could not read profile {{.*}}.nonexistent.profdata:
// CHECK-NOT: Assertion failed
