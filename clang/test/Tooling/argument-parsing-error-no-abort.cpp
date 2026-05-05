// RUN: not clang-check --nonsense %s -- 2>&1 | FileCheck %s
// CHECK: clang-check{{(\.exe)?}}: Unknown command line argument '--nonsense'

int main() { return 0; }
