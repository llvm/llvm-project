// Purpose:
//      Check that \DexExpectStepKind correctly handles recursive calls.
//      Specifically, ensure recursive calls count towards 'FUNC' and not
//      'VERTICAL_BACKWARD'.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: recursive.cpp:

int func(int i) {
    if (i > 1)
        return i + func(i - 1);
    return i;
}

int main()
{
    return func(3);
}

// main, func, func, func
// DexExpectStepKind('FUNC', 4)
// DexExpectStepKind('VERTICAL_BACKWARD', 0)
