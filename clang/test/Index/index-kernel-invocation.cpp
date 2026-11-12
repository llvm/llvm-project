// RUN: c-index-test -index-file -arch i386 -mkernel %s | FileCheck %s

// CHECK: [indexDeclaration]: kind: function | name: foobar
void foobar(void);
