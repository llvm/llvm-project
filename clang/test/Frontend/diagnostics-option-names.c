// RUN: not %clang_cc1 -fdiagnostics-show-option -Werror -Weverything %s 2> %t
// RUN: FileCheck < %t %s

int f0(int, unsigned);
int f0(int x, unsigned y) {
// CHECK: comparison of integers of different signs{{.*}} [-Werror,-Wsign-compare]
  return x < y; // expected-error {{ : 'int' and 'unsigned int'  }}
}
