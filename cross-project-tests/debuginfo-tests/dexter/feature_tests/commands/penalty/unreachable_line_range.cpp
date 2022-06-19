// Purpose:
//      Check that \DexUnreachable correctly applies a penalty if the command
//      line is stepped on.
//
// UNSUPPORTED: system-darwin
//
//
// RUN: not %dexter_regression_test -- %s | FileCheck %s
// CHECK: unreachable_line_range.cpp:

int
main()
{ // DexLabel('begin')
  return 1;
} // DexLabel('end')

// DexUnreachable(from_line=ref('begin'), to_line=ref('end'))
