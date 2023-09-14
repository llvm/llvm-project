// Test that -print-pipeline-passes works in Clang

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -print-pipeline-passes -O0 %s 2>&1 | FileCheck %s

// Don't try to check all passes, just a few to make sure that something is
// actually printed.
// CHECK: always-inline
// CHECK-SAME: annotation-remarks
void Foo(void) {}
