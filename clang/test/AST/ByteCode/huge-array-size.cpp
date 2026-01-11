// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fsyntax-only %s
// RUN: %clang_cc1 -fsyntax-only %s

// This test checks that we don't crash when encountering arrays with
// sizes that exceed the bytecode interpreter's limits.
// See: https://github.com/llvm/llvm-project/issues/175293

char q[-2U];

void foo() { char *p = q + 1; }
