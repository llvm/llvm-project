// Purpose:
//      Test that \DexFinishTest can be used with a condition, so the test exits
//      when the line referenced by \DexFinishTest is stepped on and the given
//      condition (x == 5) is satisfied.
//      Tests using the default controller (no \DexLimitSteps).
//
// REQUIRES: system-linux
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: default_conditional.cpp

int main() {
    for (int x = 0; x < 10; ++x)
        (void)0; // DexLabel('finish_line')
}

// DexFinishTest('x', 5, on_line=ref('finish_line'))
// DexExpectWatchValue('x', 0, 1, 2, 3, 4, 5)
