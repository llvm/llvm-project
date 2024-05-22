// This test started failing recently for unknown reasons.
// XFAIL:*
// RUN: %dexter_regression_test_build \
// RUN:     -fdebug-prefix-map=%S=/changed %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t \
// RUN:     --debugger %dexter_regression_test_debugger \
// RUN:     --source-root-dir=%S --debugger-use-relative-paths -- %s

#include <stdio.h>
int main() {
  int x = 42;
  printf("hello world: %d\n", x); // DexLabel('check')
}

// DexExpectWatchValue('x', 42, on_line=ref('check'))
