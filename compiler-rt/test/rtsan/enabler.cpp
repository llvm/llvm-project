// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx %s -fsanitize=realtime -o - -S -emit-llvm | FileCheck %s --check-prefix=CHECK-ENABLED-IR
// RUN: %clangxx %s -o - -S -emit-llvm | FileCheck %s --check-prefix=CHECK-DISABLED-IR
// UNSUPPORTED: ios

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/rtsan_interface.h"

void violation() {
  void *ptr = malloc(2);
  ptr = malloc(2);
  printf("ptr: %p\n", ptr); // ensure we don't optimize out the malloc

  {
    __rtsan::ScopedEnabler rtsan{};
    free(ptr);
  }
}

int main() {
  violation();
  return 0;
  // CHECK: {{.*Real-time violation.*}}
  // CHECK-NOT: {{.*malloc*}}
  // CHECK: {{.*free*}}
}

// CHECK-ENABLED-IR: {{.*@__rtsan_realtime_enter.*}}
// CHECK-ENABLED-IR: {{.*@__rtsan_realtime_exit.*}}

// CHECK-DISABLED-IR-NOT: {{.*__rtsan_realtime_enter.*}}
// CHECK-DISABLED-IR-NOT: {{.*__rtsan_realtime_exit.*}}
