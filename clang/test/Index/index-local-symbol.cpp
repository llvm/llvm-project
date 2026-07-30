void ff() {
  struct Foo {};
}

// RUN: env CINDEXTEST_INDEXLOCALSYMBOLS=1 c-index-test -index-file %s | FileCheck %s
// CHECK: [indexDeclaration]: kind: struct | name: Foo | {{.*}} | loc: 2:10