// Purpose:
//      Test that a \DexDeclareAddress command can be passed 'hit_count' as an
//      optional keyword argument that captures the value of the given
//      expression after the target line has been stepped on a given number of
//      times.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: address_hit_count.cpp

int main() {
    int *x = new int[3];
    for (int *y = x; y < x + 3; ++y)
      *y = 0; // DexLabel('test_line')
    delete x;
}

// DexDeclareAddress('y', 'y', on_line=ref('test_line'), hit_count=2)
// DexExpectWatchValue('y', address('y', -8), address('y', -4), address('y'), on_line=ref('test_line'))
