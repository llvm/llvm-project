// Test that we print pass structure with new and legacy PM.
// RUN: %clang -fdebug-pass-structure -fintegrated-as -O3 -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --strict-whitespace
// REQUIRES: asserts

// should have proper indentation, should not print any analysis information
// CHECK-NOT: Running analysis
// CHECK: {{^}}Running{{.*}}GlobalOptPass
// CHECK: {{^}}  Running{{.*}}RequireAnalysisPass{{.*}}GlobalsAA
// CHECK: GlobalOptPass
// CHECK-NOT: Invalidating analysis

void f(void) {}
