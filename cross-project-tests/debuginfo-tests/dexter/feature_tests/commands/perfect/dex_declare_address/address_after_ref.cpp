// Purpose:
//      Test that a \DexDeclareAddress value can have its value defined after
//      the first reference to that value.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: address_after_ref.cpp

int main() {
    int *x = new int(5);
    int *y = x; // DexLabel('first_line')
    delete x; // DexLabel('last_line')
}

// DexDeclareAddress('y', 'y', on_line=ref('last_line'))
// DexExpectWatchValue('x', address('y'), on_line=ref('first_line'))
