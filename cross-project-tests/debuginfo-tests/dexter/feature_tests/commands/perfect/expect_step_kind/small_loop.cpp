// Purpose:
//      Check that \DexExpectStepKind correctly counts function calls in loops
//      where the last source line in the loop is a call. Expect steps out
//      of a function to a line before the call to count as 'VERTICAL_BACKWARD'.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: small_loop.cpp:

int func(int i){
    return i;
}

int main()
{
    for (int i = 0; i < 2; ++i) {
        func(i);
    }
    return 0;
}

// DexExpectStepKind('VERTICAL_BACKWARD', 2)
