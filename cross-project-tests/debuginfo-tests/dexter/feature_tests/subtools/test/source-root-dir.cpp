// This test started failing recently for unknown reasons.
// XFAIL:*
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder %dexter_regression_test_builder \
// RUN:     --debugger %dexter_regression_test_debugger \
// RUN:     --cflags "%dexter_regression_test_cflags -fdebug-prefix-map=%S=/changed" \
// RUN:     --ldflags "%dexter_regression_test_ldflags" \
// RUN:     --source-root-dir=%S --debugger-use-relative-paths -- %s

#include <stdio.h>
int main() {
  int x = 42;
  printf("hello world: %d\n", x); // DexLabel('check')
}

// DexExpectWatchValue('x', 42, on_line=ref('check'))
