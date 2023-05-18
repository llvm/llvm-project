// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x ast -ast-print %t | FileCheck %s

int f() {
  // CHECK: for (int i = 0; int x = i < 2 ? 1 : 0; i++) {
  for (int i = 0; int x = i < 2 ? 1 : 0; i++) {
    return x;
  }
  return 0;
}

