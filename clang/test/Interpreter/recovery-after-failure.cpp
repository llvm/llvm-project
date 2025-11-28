// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s

// Failed materialization shouldn't poison subsequent statements
extern "C" int undefined_function();
int result = undefined_function();
// CHECK: error: Failed to materialize symbols

int x = 42;
// CHECK-NOT: error: Failed to materialize symbols

%quit
