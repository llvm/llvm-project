// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx %s -fsanitize=realtime -o - -S -emit-llvm | FileCheck %s --check-prefix=CHECK-ENABLED-IR
// RUN: %clangxx %s -o - -S -emit-llvm | FileCheck %s --check-prefix=CHECK-DISABLED-IR
// UNSUPPORTED: ios

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/rtsan_interface.h"

void violation() [[clang::nonblocking]] {
  void *ptr;
  {
    __rtsan::ScopedDisabler disabler{};
    ptr = malloc(2);
    printf("ptr: %p\n", ptr); // ensure we don't optimize out the malloc
  }

  free(ptr);
}

int main() {
  violation();
  return 0;
  // CHECK: {{.*Real-time violation.*}}
  // CHECK-NOT: {{.*malloc*}}
  // CHECK: {{.*free*}}
}

// CHECK-ENABLED-IR: {{.*@__rtsan_disable.*}}
// CHECK-ENABLED-IR: {{.*@__rtsan_enable.*}}

// CHECK-DISABLED-IR-NOT: {{.*__rtsan_disable.*}}
// CHECK-DISABLED-IR-NOT: {{.*__rtsan_enable.*}}
