// RUN: not clang-query --invalid-arg 2>&1 | FileCheck %s

// CHECK: error: [CommonOptionsParser]: clang-query: Unknown command line argument '--invalid-arg'.  Try: 'clang-query --help'
// CHECK-NEXT: clang-query: Did you mean '--extra-arg'?
