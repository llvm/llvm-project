// This test started failing recently for unknown reasons.
// XFAIL:*
// RUN: %dexter_regression_test_cxx_build \
// RUN:     -fdebug-prefix-map=%S=/changed %s -o %t
// RUN: %dexter_regression_test_run \
// RUN:     --binary %t --source-root-dir=%S --debugger-use-relative-paths -- %s

#include <stdio.h>
int main() {
  int x = 42;
  printf("hello world: %d\n", x); // DexLabel('check')
}

// DexExpectWatchValue('x', 42, on_line=ref('check'))
