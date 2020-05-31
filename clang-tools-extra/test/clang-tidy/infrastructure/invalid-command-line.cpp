// RUN: not clang-tidy --invalid-arg 2>&1 | FileCheck %s

// CHECK: error: [CommonOptionsParser]: clang-tidy: Unknown command line argument '--invalid-arg'.  Try: 'clang-tidy --help'
// CHECK-NEXT: clang-tidy: Did you mean '--extra-arg'?
