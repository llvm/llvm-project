// Purpose:
//      Test that \DexFinishTest can be used without a condition or hit_count,
//      so the test simply exits as soon as the line referenced by \DexFinishTest
//      is stepped on.
//      Tests using the default controller (no \DexLimitSteps).
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: default_simple.cpp

int main() {
    int x = 0;
    x = 1; // DexLabel('start_line')
    x = 2; // DexLabel('finish_line')
}

// DexFinishTest(on_line=ref('finish_line'))
// DexExpectWatchValue('x', 0, 1, from_line=ref('start_line'), to_line=ref('finish_line'))
