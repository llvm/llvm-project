// Purpose:
//      Test that when a \DexDeclareAddress never resolves to a value, it is
//      counted as a missing value in any \DexExpectWatchValues.
//
// REQUIRES: system-linux
//
// RUN: not %dexter_regression_test -- %s | FileCheck %s
// CHECK: missing_dex_address.cpp

int main() {
    int *x = nullptr;
    x = new int(5); // DexLabel('start_line')
    if (false) {
        (void)0; // DexLabel('unreachable')
    }
    delete x; // DexLabel('end_line')
}

// DexDeclareAddress('x', 'x', on_line=ref('unreachable'))
// DexExpectWatchValue('x', 0, address('x'), from_line=ref('start_line'), to_line=ref('end_line'))
