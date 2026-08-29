// REQUIRES: host-supports-jit
// RUN: clang-repl -Xcc -fno-color-diagnostics -Xcc -fno-delayed-template-parsing < %s 2>&1 | FileCheck %s

template <typename T> T my_pow(T a, T b) { return a * b; }

(10-)*my_pow(2, 2);
// CHECK: error: expected expression
// CHECK: error: Parsing failed.

int x = my_pow(2, 2);
// CHECK-NOT: JIT session error
// CHECK-NOT: Failed to materialize symbols
