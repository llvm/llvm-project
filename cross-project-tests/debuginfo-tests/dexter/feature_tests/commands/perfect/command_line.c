// The dbgeng driver doesn't support \DexCommandLine yet.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test_c_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck --dump-input-context=999999999 %s
// CHECK: command_line.c:

int main(int argc, const char **argv) {
  if (argc == 4)
    return 0; // DexLabel('retline')

  return 1; // DexUnreachable()
}

// DexExpectWatchValue('argc', '4', on_line=ref('retline'))

// Three args will be appended to the 'default' argument.
// DexCommandLine(['a', 'b', 'c'])
