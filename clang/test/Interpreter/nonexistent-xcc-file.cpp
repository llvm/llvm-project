// RUN: not clang-repl --Xcc=%t/nonexistent.cpp 2>&1 | FileCheck --ignore-case %s

// CHECK: error: error reading '{{.*}}nonexistent.cpp': No such file or directory
// CHECK-NOT: Compiler instance not registered
// CHECK-NOT: Assertion
// CHECK-NOT: segmentation fault
// CHECK-NOT: core dumped
