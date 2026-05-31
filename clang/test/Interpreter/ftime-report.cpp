// Tests that -ftime-report works with clang-repl without crashing.
// RUN: clang-repl -Xcc -ftime-report "int x = 42;" 2>&1 | FileCheck %s
// CHECK-NOT: Assertion
// CHECK-NOT: PLEASE submit a bug report
// CHECK: Clang time report
