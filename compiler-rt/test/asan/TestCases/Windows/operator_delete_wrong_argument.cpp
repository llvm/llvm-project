// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <windows.h>

int main() {
  int *x = new int[42];
  delete (x + 1);
// CHECK: AddressSanitizer: attempting free on address which was not malloc()-ed
// CHECK:   {{#0 0x.* operator delete}}
// CHECK:   {{#1 .* main .*operator_delete_wrong_argument.cpp}}:[[@LINE-3]]
}
