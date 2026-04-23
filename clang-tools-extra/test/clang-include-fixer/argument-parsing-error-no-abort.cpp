// RUN: not clang-include-fixer --nonsense %s -- 2>&1 | FileCheck %s
// CHECK: clang-include-fixer: Unknown command line argument '--nonsense'

int main() { return 0; }
