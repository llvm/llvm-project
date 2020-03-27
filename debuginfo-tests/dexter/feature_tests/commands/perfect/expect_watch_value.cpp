// Purpose:
//      Check that \DexExpectWatchValue applies no penalties when expected
//      values are found.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
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
DexExpectWatchValue('n', '8', on_line='entry')
DexExpectWatchValue('i',
                    '1', '2', '3', '4', '5', '6', '7', '8',
                    on_line='loop')

DexExpectWatchValue('fac',
                    '1', '2', '6', '24', '120', '720', '5040',
                     on_line='loop')

DexExpectWatchValue('n', '8', on_line='loop')
DexExpectWatchValue('fac', '40320', on_line='ret')
DexExpectWatchValue('n', '8', on_line='ret')
*/
