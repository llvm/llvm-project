// Purpose:
//      Test that a \DexDeclareAddress value can be used to compare two equal
//      pointer variables.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: identical_address.cpp

int main() {
    int *x = new int(5);
    int *y = x;
    delete x; // DexLabel('test_line')
}

// DexDeclareAddress('x', 'x', on_line=ref('test_line'))
// DexExpectWatchValue('x', address('x'), on_line=ref('test_line'))
// DexExpectWatchValue('y', address('x'), on_line=ref('test_line'))
