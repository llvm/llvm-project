// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s

// Verify that clang-repl recovers cleanly after a deferred template
// instantiation error. The failed cell must not contaminate the CodeGen module
// used by subsequent cells.

template <typename T> T f(T a) { static_assert(sizeof(T) == 0, "unsupported type"); return a; }

f(1.0);
// CHECK: error: static assertion failed

int x = 10;
x
// CHECK: (int) 10

%quit
