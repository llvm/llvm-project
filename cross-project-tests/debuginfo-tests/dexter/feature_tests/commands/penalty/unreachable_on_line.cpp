// Purpose:
//      Check that \DexUnreachable correctly applies a penalty if the command
//      line is stepped on.
//
// UNSUPPORTED: system-darwin
//
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: unreachable_on_line.cpp:

int
main()
{
  return 1;  // DexLabel('this_one')
}

// DexUnreachable(on_line=ref('this_one'))
