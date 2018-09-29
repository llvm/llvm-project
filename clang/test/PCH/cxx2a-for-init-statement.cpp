// RUN: %clang_cc1 -emit-pch -std=c++2a -o %t %s
// RUN: %clang_cc1 -std=c++2a -x ast -ast-print %t | FileCheck %s

void f() {
  // CHECK:      for (int arr[3]; int n : arr) {
  // CHECK-NEXT: }
  for (int arr[3]; int n : arr) {}
}
