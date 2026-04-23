// RUN: not clang-change-namespace --nonsense %s -- 2>&1 | FileCheck %s
// CHECK: clang-change-namespace: Unknown command line argument '--nonsense'

int main() { return 0; }
