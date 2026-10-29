// RUN: not clang-refactor --nonsense 2>&1 | FileCheck %s
// CHECK: clang-refactor{{(\.exe)?}}: Unknown command line argument '--nonsense'
