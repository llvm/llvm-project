// Purpose:
//      Test that a \DexDeclareAddress value can be used to check the change in
//      value of a variable over time, relative to its initial value.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: self_comparison.cpp

int main() {
    int *x = new int[3];
    for (int *y = x; y < x + 3; ++y)
      *y = 0; // DexLabel('test_line')
    delete x;
}

// DexDeclareAddress('y', 'y', on_line=ref('test_line'))
// DexExpectWatchValue('y', address('y'), address('y', 4), address('y', 8), on_line=ref('test_line'))
