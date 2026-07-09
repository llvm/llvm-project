// RUN: not clang-move --nonsense %s -- 2>&1 | FileCheck %s
// CHECK: clang-move{{(\.exe)?}}: Unknown command line argument '--nonsense'

int main() { return 0; }
