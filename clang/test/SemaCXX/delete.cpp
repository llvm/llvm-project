// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: %clang_cc1 -fixit -x c++ %t
// RUN: %clang_cc1 -E -o - %t | FileCheck %s

void f(int a[10][20]) {
  // CHECK: delete[] a;
  delete a; // expected-warning {{'delete' applied to a pointer-to-array type}}
}
