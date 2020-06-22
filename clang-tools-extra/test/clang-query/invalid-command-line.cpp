// RUN: not clang-query --invalid-arg 2>&1 | FileCheck %s

// CHECK: error: [CommonOptionsParser]: clang-query{{(\.exe)?}}: Unknown command line argument '--invalid-arg'.  Try: 'clang-query{{(\.exe)?}} --help'
// CHECK-NEXT: clang-query{{(\.exe)?}}: Did you mean '--extra-arg'?
