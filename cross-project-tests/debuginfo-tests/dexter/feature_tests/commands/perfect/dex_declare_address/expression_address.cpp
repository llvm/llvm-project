// Purpose:
//      Test that a \DexDeclareAddress value can be used to compare the
//      addresses of two local variables that refer to the same address.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: expression_address.cpp

int main() {
    int x = 5;
    int &y = x;
    x = 3; // DexLabel('test_line')
}

// DexDeclareAddress('x_addr', '&x', on_line=ref('test_line'))
// DexExpectWatchValue('&x', address('x_addr'), on_line=ref('test_line'))
// DexExpectWatchValue('y', address('x_addr'), on_line=ref('test_line'))
