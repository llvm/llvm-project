// RUN: %clang_cc1 -Wall -Wextra -Wuninitialized -fsyntax-only %s 2>&1 | FileCheck %s

void test1() {
  int arr[5];
  int x = arr[0]; // expected-warning{{variable 'arr' is uninitialized when used here}}
}

void test2() {
  int a[3][3];
  int y = a[1][1]; // expected-warning{{variable 'a' is uninitialized when used here}}
}

void test3() {
  int n;
  int vla[n]; // expected-note{{declared here}}
  int z = vla[2]; // expected-warning{{variable 'vla' is uninitialized when used here}}
}
