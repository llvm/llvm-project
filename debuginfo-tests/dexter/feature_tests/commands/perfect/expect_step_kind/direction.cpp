// Purpose:
//      Check that \DexExpectStepKind correctly counts 'VERTICAL_BACKWARD' steps
//      for a trivial test. Expect one 'VERTICAL_BACKWARD' for every step onto
//      a lesser source line number in the same function. Expect one
//      'VERTICAL_FORWARD' for every step onto a greater source line number in
//      the same function.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w  \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: direction.cpp:

int func(int i) {
    return i; // step 7, 9, 11
}

int main()
{
    for (int i = 0; i < 2; ++i) { // step 1: FUNC, step 3, 5: VERTICAL_BACKWARD
        i = i;                    // step 2, 4: VERTICAL_FORWARD
    }
    // ---------1           - step 6: VERTICAL_FORWARD
    // ---------|---------2 - step 8: HORIZONTAL_FORWARD
    // ----3----|---------| - step 10: HORIZONTAL_BACKWARD
    return func(func(0) + func(1));
}

// DexExpectStepKind('VERTICAL_BACKWARD', 2)
// DexExpectStepKind('VERTICAL_FORWARD', 3)
// DexExpectStepKind('HORIZONTAL_FORWARD', 1)
// DexExpectStepKind('HORIZONTAL_BACKWARD', 1)
