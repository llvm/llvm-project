// Purpose:
//    Check that \DexUnreachable has no effect if the command line is never
//    stepped on.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: unreachable.cpp:

int main()
{
  return 0;
  return 1; // DexUnreachable()
}
