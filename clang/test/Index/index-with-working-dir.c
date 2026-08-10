
void foo();

// RUN: c-index-test -index-file -working-directory=%S %s | FileCheck %s
// CHECK: [indexDeclaration]: kind: function | name: foo
