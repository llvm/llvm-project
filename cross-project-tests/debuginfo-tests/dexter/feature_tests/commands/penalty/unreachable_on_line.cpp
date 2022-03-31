// Purpose:
//      Check that \DexUnreachable correctly applies a penalty if the command
//      line is stepped on.
//
// UNSUPPORTED: system-darwin
//
//
// RUN: not %dexter_regression_test -- %s | FileCheck %s
// CHECK: unreachable_on_line.cpp:

int
main()
{
  return 1;  // DexLabel('this_one')
}

// DexUnreachable(on_line=ref('this_one'))
