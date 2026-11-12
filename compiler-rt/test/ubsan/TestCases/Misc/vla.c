// RUN: %clang -fsanitize=vla-bound %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MINUS-ONE
// RUN: %run %t a 2>&1 | FileCheck %s --check-prefix=CHECK-ZERO
// RUN: %run %t a b

int main(int argc, char **argv) {
  // CHECK-MINUS-ONE: vla.c:[[@LINE+2]]:11: runtime error: variable length array bound evaluates to non-positive value -1
  // CHECK-ZERO: vla.c:[[@LINE+1]]:11: runtime error: variable length array bound evaluates to non-positive value 0
  int arr[argc - 2];
  return 0;
}
