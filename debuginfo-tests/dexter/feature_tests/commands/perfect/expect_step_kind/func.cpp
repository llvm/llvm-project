// Purpose:
//      Check that \DexExpectStepKind correctly counts 'FUNC' steps for a
//      trivial test. Expect one 'FUNC' per call to a function which is defined
//      in one of the source files in the test directory.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: func.cpp:

int func(int i) {
    return i;
}

int main()
{
    func(0);
    func(1);
    return 0;
}

// main, func, func
// DexExpectStepKind('FUNC', 3)
