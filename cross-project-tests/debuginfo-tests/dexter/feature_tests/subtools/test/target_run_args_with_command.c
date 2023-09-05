// The dbgeng driver doesn't support --target-run-args yet.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test --target-run-args "a b 'c d'" -- %s | FileCheck %s
// CHECK: target_run_args_with_command.c:

int main(int argc, const char **argv) {
  if (argc == 6)
    return 0; // DexLabel('retline')

  return 1; // DexUnreachable()
}

// DexCommandLine(['e', 'f'])
// DexExpectWatchValue('argc', '6', on_line=ref('retline'))
// DexExpectWatchValue('argv[1][0]', "'e'", on_line=ref('retline'))
