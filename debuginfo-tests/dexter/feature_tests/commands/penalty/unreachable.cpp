// Purpose:
//      Check that \DexUnreachable correctly applies a penalty if the command
//      line is stepped on.
//
// REQUIRES: system-linux, lldb
//
// RUN: not %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: unreachable.cpp:

int
main()
{
  return 1;  // DexUnreachable()
}
