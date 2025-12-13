// The dbgeng driver doesn't support --target-run-args yet.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test_c_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t --target-run-args "a b 'c d'" -- %s | FileCheck %s
// CHECK: target_run_args.c:

int main(int argc, const char **argv) {
  if (argc == 4)
    return 0; // DexLabel('retline')

  return 1; // DexUnreachable()
}

// DexExpectWatchValue('argc', '4', on_line=ref('retline'))
// DexExpectWatchValue('argv[1][0]', "'a'", on_line=ref('retline'))
