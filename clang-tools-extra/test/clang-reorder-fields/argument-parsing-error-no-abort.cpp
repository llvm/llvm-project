// RUN: not clang-reorder-fields --nonsense %s -- 2>&1 | FileCheck %s
// CHECK: clang-reorder-fields: Unknown command line argument '--nonsense'

int main() { return 0; }
