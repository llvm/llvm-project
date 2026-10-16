// RUN: not find-all-symbols --nonsense %s -- 2>&1 | FileCheck %s
// CHECK: find-all-symbols{{(\.exe)?}}: Unknown command line argument '--nonsense'

int main() { return 0; }
