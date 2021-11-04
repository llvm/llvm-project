// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>

namespace __tsan {

void OnPotentiallyBlockingRegionBegin() {
  printf("Enter __cxa_guard_acquire\n");
}

void OnPotentiallyBlockingRegionEnd() { printf("Exit __cxa_guard_acquire\n"); }

} // namespace __tsan

int main(int argc, char **argv) {
  // CHECK: Enter main
  printf("Enter main\n");
  // CHECK-NEXT: Enter __cxa_guard_acquire
  // CHECK-NEXT: Exit __cxa_guard_acquire
  static int s = argc;
  (void)s;
  // CHECK-NEXT: Exit main
  printf("Exit main\n");
  return 0;
}
