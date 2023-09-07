// The dbgeng driver doesn't support \DexCommandLine yet.
// UNSUPPORTED: system-windows
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: command_line.c:

int main(int argc, const char **argv) {
  if (argc == 4)
    return 0; // DexLabel('retline')

  return 1; // DexUnreachable()
}

// DexExpectWatchValue('argc', '4', on_line=ref('retline'))

// Three args will be appended to the 'default' argument.
// DexCommandLine(['a', 'b', 'c'])
