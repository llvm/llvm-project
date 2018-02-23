// RUN: %clangxx -fsanitize=bounds %s -O3 -o %t
// RUN: %run %t 0 0 0
// RUN: %run %t 1 2 3
// RUN: %expect_crash %run %t 2 0 0 2>&1 | FileCheck %s --check-prefix=CHECK-A-2
// RUN: %run %t 0 3 0 2>&1 | FileCheck %s --check-prefix=CHECK-B-3
// RUN: %run %t 0 0 4 2>&1 | FileCheck %s --check-prefix=CHECK-C-4

int main(int argc, char **argv) {
  int arr[2][3][4] = {};

  return arr[argv[1][0] - '0'][argv[2][0] - '0'][argv[3][0] - '0'];
  // CHECK-A-2: bounds.cpp:[[@LINE-1]]:10: runtime error: index 2 out of bounds for type 'int [2][3][4]'
  // CHECK-B-3: bounds.cpp:[[@LINE-2]]:10: runtime error: index 3 out of bounds for type 'int [3][4]'
  // CHECK-C-4: bounds.cpp:[[@LINE-3]]:10: runtime error: index 4 out of bounds for type 'int [4]'
}
