// Purpose:
//    Check that \DexUnreachable doesn't trigger on the line it's specified
//    on, if it has a specifier indicating which lines should be unreachable.
//
// UNSUPPORTED: system-darwin
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: unreachable_not_cmd_lineno.cpp:

int main(int argc, char **argv)
{
  if (argc != 1)
    return 1; // DexLabel('this_one')
  else
    return 0; // DexUnreachable(on_line=ref('this_one')) DexUnreachable(from_line=ref('this_one'), to_line=ref('this_one'))
}

