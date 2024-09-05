// Trivial check to ensure skip-function-bodies flag is propagated.
//
// RUN: %clang_cc1 -verify -skip-function-bodies -pedantic-errors %s
// expected-no-diagnostics

int f() {
  // normally this should emit some diags, but we're skipping it!
  this is garbage;
}

// Make sure we only accept it as a cc1 arg.
// RUN: not %clang -skip-function-bodies %s 2>&1 | FileCheck %s
// CHECK: clang: error: unknown argument '-skip-function-bodies'; did you mean '-Xclang -skip-function-bodies'?
