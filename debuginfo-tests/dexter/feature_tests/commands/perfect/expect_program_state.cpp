// Purpose:
//      Check that \DexExpectWatchValue applies no penalties when expected
//      program states are found.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -glldb" -- %s \
// RUN:     | FileCheck %s
// CHECK: expect_program_state.cpp:

int GCD(int lhs, int rhs)
{
    if (rhs == 0)
        return lhs; // DexLabel('check')
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
                'lhs': '37', 'rhs': '0'
            }
        },
        {
            'watches': {
                'lhs': {'value': '111'}, 'rhs': {'value': '37'}
            }
        },
        {
            'watches': {
                'lhs': {'value': '259'}, 'rhs': {'value': '111'}
            }
        },
        {
            'watches': {
                'lhs': {'value': '111'}, 'rhs': {'value': '259'}
            }
        }
    ]
})
*/
