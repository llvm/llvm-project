// Purpose:
//      Check that \DexExpectWatchValue correctly applies a penalty when
//      expected values are not found.
//
// UNSUPPORTED: system-darwin
//
//
// RUN: not %dexter_regression_test -- %s | FileCheck %s
// CHECK: expect_watch_value.cpp:

int main()
{
    for (int i = 0; i < 3; ++i)
        int a = i; // DexLabel('loop')
    return 0;  // DexLabel('ret')
}

// DexExpectWatchValue('i', '0', '1', '2', on_line='loop')
// DexExpectWatchValue('i', '3', on_line='ret')
// ---------------------^ out of scope
