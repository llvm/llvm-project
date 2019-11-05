// Purpose:
//    Check that \DexUnreachable has no effect if the command line is never
//    stepped on.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: unreachable.cpp:

int main()
{
  return 0;
  return 1; // DexUnreachable()
}
