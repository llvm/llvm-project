// Instrumentation from PGO/code coverage isn't supported yet for dyndbg.
// RUN: not %clang_cc1  %s -fprofile-instrument=clang -fdynamic-debugging -emit-llvm \
// RUN:    --discard-dynamic-debugging-debug-module 2>&1 \
// RUN: | FileCheck %s
// CHECK: error: '-fdynamic-debugging' unsupported with instrumentation (PGO/code coverage)

int b() { return 0; }
