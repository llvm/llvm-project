// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int *p;

int main() {
  for (int i = 0; i < 3; i++) {
    int x;
    p = &x;
  }
  return *p;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope-loop-removed.cpp:[[@LINE-2]]
  // CHECK: Address 0x{{.*}} is located in stack of thread T{{.*}} at offset [[OFFSET:[^ ]+]] in frame
  // {{\[}}[[OFFSET]], {{[0-9]+}}) 'x'
}
