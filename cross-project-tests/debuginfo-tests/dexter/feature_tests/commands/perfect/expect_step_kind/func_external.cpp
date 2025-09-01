// Purpose:
//      Check that \DexExpectStepKind correctly counts 'FUNC_EXTERNAL' steps
//      for a trivial test. Expect one 'FUNC_EXTERNAL' per external call.
//
// UNSUPPORTED: system-darwin
//
// XFAIL:*
// This fails right now on my linux and windows machine, needs examining as to
// why.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: func_external.cpp:

#include <cstdlib>

int func(int i){
    return abs(i);
}

int main()
{
    func(0);
    func(1);
    return 0;
}

// DexExpectStepKind('FUNC_EXTERNAL', 2)
