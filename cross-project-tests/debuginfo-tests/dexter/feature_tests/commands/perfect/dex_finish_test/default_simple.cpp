// Purpose:
//      Test that \DexFinishTest can be used without a condition or hit_count,
//      so the test simply exits as soon as the line referenced by \DexFinishTest
//      is stepped on.
//      Tests using the default controller (no \DexLimitSteps).
//
// REQUIRES: system-linux
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: default_simple.cpp

int main() {
    int x = 0; // DexLabel('start_line')
    x = 1;
    x = 2; // DexLabel('finish_line')
}

// DexFinishTest(on_line=ref('finish_line'))
// DexExpectWatchValue('x', 0, 1)
