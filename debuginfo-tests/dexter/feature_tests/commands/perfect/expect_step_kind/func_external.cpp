// Purpose:
//      Check that \DexExpectStepKind correctly counts 'FUNC_EXTERNAL' steps
//      for a trivial test. Expect one 'FUNC_EXTERNAL' per external call.
//
// REQUIRES: system-linux, lldb
// XFAIL: system-linux
// This fails right now on my linux machine, needs examining as to why.
//
// RUN: %dexter --fail-lt 1.0 -w  \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
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
