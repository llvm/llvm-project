// Purpose:
//      Check that \DexExpectWatchValue applies no penalties when expected
//      values are found.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: expect_watch_value.cpp:

unsigned long Factorial(int n) {
    volatile unsigned long fac = 1; // DexLabel('entry')

    for (int i = 1; i <= n; ++i)
        fac *= i;                   // DexLabel('loop')

    return fac;                     // DexLabel('ret')
}

int main()
{
    return Factorial(8);
}

/*
DexExpectWatchValue('n', '8', on_line=ref('entry'))
DexExpectWatchValue('i',
                    '1', '2', '3', '4', '5', '6', '7', '8',
                    on_line=ref('loop'))

DexExpectWatchValue('fac',
                    '1', '2', '6', '24', '120', '720', '5040',
                     on_line=ref('loop'))

DexExpectWatchValue('n', '8', on_line=ref('loop'))
DexExpectWatchValue('fac', '40320', on_line=ref('ret'))
DexExpectWatchValue('n', '8', on_line=ref('ret'))
*/
