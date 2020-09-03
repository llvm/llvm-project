// Purpose:
//    Check that \DexUnreachable has no effect if the command line is never
//    stepped on.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: unreachable.cpp:

int main()
{
  return 0;
  return 1; // DexUnreachable()
}
