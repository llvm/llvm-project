// Test that we print pass structure with new and legacy PM.
// RUN: %clang -fexperimental-new-pass-manager -fdebug-pass-structure -fintegrated-as -O3 -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM --strict-whitespace
// RUN: %clang -flegacy-pass-manager -fdebug-pass-structure -O0 -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEGACYPM
// REQUIRES: asserts

// should have proper indentation, should not print any analysis information
// NEWPM-NOT: Running analysis
// NEWPM: {{^}}Running{{.*}}GlobalOptPass
// NEWPM: {{^}}  Running{{.*}}RequireAnalysisPass{{.*}}GlobalsAA
// NEWPM: GlobalOptPass
// NEWPM-NOT: Invalidating analysis

// LEGACYPM: Pass Arguments:

void f() {}
