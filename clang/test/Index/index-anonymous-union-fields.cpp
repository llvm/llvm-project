struct X {
  union {
    void *a;
  };
};

// RUN: c-index-test -index-file %s > %t
// RUN: FileCheck %s -input-file=%t

// CHECK: [indexDeclaration]: kind: field | name: a | {{.*}} | loc: 3:11
