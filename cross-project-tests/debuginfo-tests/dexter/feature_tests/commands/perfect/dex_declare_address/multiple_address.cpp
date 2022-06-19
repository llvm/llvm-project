// Purpose:
//      Test that multiple \DexDeclareAddress references that point to different
//      addresses can be used within a single \DexExpectWatchValue.
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: multiple_address.cpp

int main() {
    int *x = new int(5);
    int *y = new int(4);
    int *z = x;
    *z = 0; // DexLabel('start_line')
    z = y;
    *z = 0;
    delete x; // DexLabel('end_line')
    delete y;
}

// DexDeclareAddress('x', 'x', on_line=ref('start_line'))
// DexDeclareAddress('y', 'y', on_line=ref('start_line'))
// DexExpectWatchValue('z', address('x'), address('y'), from_line=ref('start_line'), to_line=ref('end_line'))
// DexExpectWatchValue('*z', 5, 0, 4, 0, from_line=ref('start_line'), to_line=ref('end_line'))
