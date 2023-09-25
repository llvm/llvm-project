// REQUIRES: system-windows
//
// RUN: %clang_cl /Z7 /Zi %s -o %t
// RUN: %dexter --fail-lt 1.0 -w --binary %t --debugger 'dbgeng' -- %s

#include <stdio.h>
int main() {
  printf("hello world\n");
  int x = 42;
  __debugbreak(); // DexLabel('stop')
}

// DexExpectWatchValue('x', 42, on_line=ref('stop'))
