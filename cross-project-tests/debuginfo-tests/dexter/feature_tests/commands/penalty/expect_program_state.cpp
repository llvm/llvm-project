// Purpose:
//      Check that \DexExpectProgramState correctly applies a penalty when
//      an expected program state is never found.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: expect_program_state.cpp:

int GCD(int lhs, int rhs)
{
    if (rhs == 0)   // DexLabel('check')
        return lhs;
    return GCD(rhs, lhs % rhs);
}

int main()
{
    return GCD(111, 259);
}

/*
DexExpectProgramState({
    'frames': [
        {
            'location': {
                'lineno': ref('check')
            },
            'watches': {
                'lhs': '0', 'rhs': '0'
            }
        },
    ]
})
*/
