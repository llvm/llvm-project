// REQUIRES: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w --builder 'clang-cl_vs2015' \
// RUN:      --debugger 'dbgeng' --cflags '/Z7 /Zi' --ldflags '/Z7 /Zi' -- %s

#include <stdio.h>
int main() {
  printf("hello world\n");
  int x = 42;
  __debugbreak(); // DexLabel('stop')
}

// DexExpectWatchValue('x', 42, on_line='stop')
