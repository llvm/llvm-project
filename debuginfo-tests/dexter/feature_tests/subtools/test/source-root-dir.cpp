// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-O0 -glldb -fdebug-prefix-map=%S=/changed" \
// RUN:     --source-root-dir=%S --debugger-use-relative-paths -- %s

#include <stdio.h>
int main() {
  int x = 42;
  printf("hello world: %d\n", x); // DexLabel('check')
}

// DexExpectWatchValue('x', 42, on_line=ref('check'))
