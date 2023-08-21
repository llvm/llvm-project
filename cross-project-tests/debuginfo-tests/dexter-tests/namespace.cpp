// Purpose:
// Ensure that the debug information for a global variable includes
// namespace information.

// REQUIRES: lldb
// UNSUPPORTED: system-windows

// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-g -O0" -v -- %s

#include <stdio.h>

namespace monkey {
const int ape = 32;
}

int main() {
  printf("hello %d\n", monkey::ape); // DexLabel('main')
  return 0;
}

// DexExpectWatchValue('monkey::ape', 32, on_line=ref('main'))

