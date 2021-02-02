// RUN: not clang-tidy --invalid-arg 2>&1 | FileCheck %s

// CHECK: error: clang-tidy{{(\.exe)?}}: Unknown command line argument '--invalid-arg'.  Try: 'clang-tidy{{(\.exe)?}} --help'
// CHECK-NEXT: clang-tidy{{(\.exe)?}}: Did you mean '--extra-arg'?
