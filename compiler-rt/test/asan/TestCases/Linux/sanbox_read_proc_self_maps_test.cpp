// REQUIRES: x86_64-target-arch
// RUN: %clangxx_asan  %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  __sanitizer_sandbox_arguments args = {0};
  // should cache /proc/self/maps
  __sanitizer_sandbox_on_notify(&args);

  if (unshare(CLONE_NEWUSER)) {
    printf("unshare failed\n");
    return 1;
  }

  // remove access to /proc/self/maps
  if (chroot("/tmp")) {
    printf("chroot failed\n");
    return 2;
  }

  *(volatile int*)0x42 = 0;
// CHECK-NOT: CHECK failed
}
