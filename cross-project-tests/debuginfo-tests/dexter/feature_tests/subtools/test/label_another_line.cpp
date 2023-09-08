// Purpose:
//    Check that the optional keyword argument 'on_line' makes a \DexLabel label
//    that line instead of the line the command is found on.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: label_another_line.cpp: (1.0000)

int main() {
  int result = 0;
  return result;
}

// DexLabel('test', on_line=11)
// DexExpectWatchValue('result', '0', on_line=ref('test'))
