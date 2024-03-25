// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sanitizer/tsan_interface.h>
#include <stdio.h>

namespace __tsan {

#if (__APPLE__)
__attribute__((weak))
#endif
void OnPotentiallyBlockingRegionBegin() {
  printf("Enter __cxa_guard_acquire\n");
}

#if (__APPLE__)
__attribute__((weak))
#endif
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
