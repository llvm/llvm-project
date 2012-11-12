void foo() {
  int x;
  if (x) {
}

// RUN: c-index-test -cursor-at=%s:2:7 %s > %t
// RUN: FileCheck %s -input-file %t

// CHECK: VarDecl=x:2:7
