// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// Purpose:
//    Check that the optional keyword argument 'on_line' makes a \DexLabel label
//    that line instead of the line the command is found on.
//
// RUN: %dexter_regression_test -- %s | FileCheck %s
// CHECK: label_another_line.cpp: (1.0000)

int main() {
  int result = 0;
  return result;
}

// DexLabel('test', on_line=13)
// DexExpectWatchValue('result', '0', on_line=ref('test'))
