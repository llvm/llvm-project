struct S {
  int {
};
typedef struct S S;

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK: [indexDeclaration]: kind: struct | name: S |
// CHECK-NOT: [indexDeclaration]: kind: struct | name: S |
