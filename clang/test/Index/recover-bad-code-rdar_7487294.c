// RUN: %clang-cc1 -fsyntax-only %s 2>&1 | FileCheck %s

// IMPORTANT: This test case intentionally DOES NOT use --disable-free.  It
// tests that we are properly reclaiming the ASTs and we do not have a double free.
// Previously we tried to free the size expression of the VLA twice.

int foo(int x) {
  int y[x * 3];
  help
};

// CHECK: 9:3: error: use of undeclared identifier 'help'
// CHECK:  help
