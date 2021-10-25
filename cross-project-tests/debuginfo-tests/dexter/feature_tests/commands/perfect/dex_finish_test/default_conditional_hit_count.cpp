// Purpose:
//      Test that \DexFinishTest can be used with a combination of a hit_count
//      and a condition, so that the test exits after the line referenced
//      by \DexFinishTest is stepped on while the condition (x == 2) is true a
//      given number of times.
//      Tests using the default controller (no \DexLimitSteps).
//
// REQUIRES: system-linux
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: default_conditional_hit_count.cpp

int main() {
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            (void)0; // DexLabel('finish_line')
}

// DexFinishTest('x', 2, on_line=ref('finish_line'), hit_count=2)
// DexExpectWatchValue('x', 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, on_line=ref('finish_line'))
// DexExpectWatchValue('y', 0, 1, 2, on_line=ref('finish_line'))
