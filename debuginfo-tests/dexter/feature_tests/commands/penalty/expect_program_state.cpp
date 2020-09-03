// Purpose:
//      Check that \DexExpectProgramState correctly applies a penalty when
//      an expected program state is never found.
//
// UNSUPPORTED: system-darwin
//
// RUN: not %dexter_regression_test -- %s | FileCheck %s
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
                'lineno': 'check'
            },
            'watches': {
                'lhs': '0', 'rhs': '0'
            }
        },
    ]
})
*/
