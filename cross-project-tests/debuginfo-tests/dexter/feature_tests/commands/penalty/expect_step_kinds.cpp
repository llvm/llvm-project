// Purpose:
//      Check that \DexExpectStepKind correctly applies a penalty when
//      unexpected step kinds are encountered.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: expect_step_kinds.cpp:

int abs(int i){
    return i < 0? i * -1: i;
}

int main()
{
    volatile int x = 2;
    for (int i = 0; i < x; ++i) {
        abs(i);
    }
    return 0;
}

// DexExpectStepKind('FUNC', 5)
// DexExpectStepKind('FUNC_EXTERNAL', 2)
// DexExpectStepKind('VERTICAL_BACKWARD', 2)
